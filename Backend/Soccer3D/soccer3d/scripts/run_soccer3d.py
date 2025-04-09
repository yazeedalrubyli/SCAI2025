#!/usr/bin/env python
"""
Soccer3D Command Line Tool

This script provides a command-line interface to the Soccer3D library.
"""
# Standard library imports
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing

# Third-party imports
import numpy as np
import cv2
from ultralytics import YOLO  # Import Ultralytics YOLO

# Local imports
from soccer3d import initialize
from soccer3d.models import (
    initialize_mp_pose_pool,
    process_pose_from_detection,
    warmup_pytorch_cuda,
    warmup_yolo_models,
)
from soccer3d.utils import (
    extract_camera_position,
    extract_camera_direction,
    validate_intrinsics,
    get_ray_from_camera,
    find_ray_intersection,
    triangulate_pose,
    calculate_player_orientation,
    get_cardinal_direction,
    get_field_direction,
)
from soccer3d.utils.camera import preload_camera_frames, load_camera_transforms_cached, preload_all_frames
from soccer3d.utils.threading import get_thread_pool, shutdown_thread_pools
from soccer3d.logger import SuppressOutput

# Global variables
logger = None
config = None
# YOLO model caches - to avoid reloading models
player_model = None
ball_model = None

# Global result cache for batching writes
_OUTPUT_CACHE = []
_OUTPUT_CACHE_LOCK = threading.RLock()
_LAST_WRITE_TIME = 0
_WRITE_INTERVAL = 5.0  # Seconds between batch writes

# Global preloaded frames cache
_ALL_PRELOADED_FRAMES = {}

# Add Ultralytics YOLO functions
def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO model using Ultralytics.
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        YOLO model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
    
    try:
        # Ultralytics handles device selection automatically
        model = YOLO(model_path)
        logger.info(f"Loaded YOLO model from {model_path} using device: {model.device}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise

def perform_yolo_inference_batched(
    model_path: str,
    batch_frames: List,
    conf_threshold: float,
    config: Dict,
    preprocessed_data=None
) -> List:
    """
    Perform YOLO inference on a batch of frames using Ultralytics.
    
    Args:
        model_path: Path to the YOLO model
        batch_frames: List of frame images
        conf_threshold: Confidence threshold for detections
        config: Configuration dictionary
        preprocessed_data: Ignored for Ultralytics implementation
        
    Returns:
        List of detection results for each frame
    """
    global player_model, ball_model
    
    # Use global models to avoid reloading
    if "player" in model_path:
        if player_model is None:
            logger.info(f"Loading player YOLO model from {model_path}")
            player_model = load_yolo_model(model_path)
        model = player_model
    else:  # ball model
        if ball_model is None:
            logger.info(f"Loading ball YOLO model from {model_path}")
            ball_model = load_yolo_model(model_path)
        model = ball_model
    
    # Run inference using Ultralytics
    # We use explicit batch size instead of auto-batching to control memory usage
    MAX_INTERNAL_BATCH = 16  # To prevent OOM on lower-memory GPUs
    batch_results = []
    
    # Process in smaller internal batches if needed
    for start_idx in range(0, len(batch_frames), MAX_INTERNAL_BATCH):
        end_idx = min(start_idx + MAX_INTERNAL_BATCH, len(batch_frames))
        sub_batch = batch_frames[start_idx:end_idx]
        
        try:
            # Run inference on batch (Ultralytics handles batching internally)
            # conf is confidence threshold, verbose=False prevents output to console
            results = model(sub_batch, conf=conf_threshold, verbose=False)
            
            # Process results into the expected format
            for i, result in enumerate(results):
                # Create a detection result object similar to previous format
                class DetectionResult:
                    def __init__(self, xyxy, confidence, class_ids):
                        self.xyxy = xyxy
                        self.confidence = confidence
                        self.class_ids = class_ids
                
                # Extract bounding boxes, confidences, and class IDs
                boxes = []
                confidences = []
                class_ids = []
                
                if len(result.boxes) > 0:
                    # Extract boxes in xyxy format (xmin, ymin, xmax, ymax)
                    for box in result.boxes:
                        boxes.append(box.xyxy[0].tolist())  # Get as list
                        confidences.append(float(box.conf[0]))
                        class_ids.append(int(box.cls[0]))
                
                detection_result = DetectionResult(boxes, confidences, class_ids)
                batch_results.append([detection_result])
        
        except Exception as e:
            logger.error(f"Error processing batch with YOLO: {e}")
            # Add None for each frame in the failed batch
            batch_results.extend([None] * len(sub_batch))
    
    return batch_results

def warmup_yolo_models(config: Dict):
    """
    Warm up YOLO models by running a dummy inference.
    
    Args:
        config: Configuration dictionary
    """
    global player_model, ball_model
    
    # Warm up player model
    player_model_path = config.get('player_model_path', "soccer3d/models/player_model/model.engine")
    logger.info(f"Warming up player YOLO model at {player_model_path}")
    if not os.path.exists(player_model_path):
        logger.warning(f"Player model not found at {player_model_path}")
    else:
        # Create a dummy input
        dummy_input = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            player_model = load_yolo_model(player_model_path)
            _ = player_model(dummy_input, verbose=False)
        except Exception as e:
            logger.error(f"Error warming up player model: {e}")
    
    # Warm up ball model
    ball_model_path = config.get('ball_model_path', "soccer3d/models/ball_model/model.engine")
    logger.info(f"Warming up ball YOLO model at {ball_model_path}")
    if not os.path.exists(ball_model_path):
        logger.warning(f"Ball model not found at {ball_model_path}")
    else:
        # Create a dummy input
        dummy_input = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            ball_model = load_yolo_model(ball_model_path)
            _ = ball_model(dummy_input, verbose=False)
        except Exception as e:
            logger.error(f"Error warming up ball model: {e}")
    
    logger.info("YOLO models warmed up successfully")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Soccer3D: Soccer player and ball 3D tracking")
    
    # Required/key options
    parser.add_argument("--frame", type=int, default=160,
                        help="Frame number to process (default: 160)")
    parser.add_argument("--start-frame", type=int,
                        help="Starting frame number for processing multiple frames")
    parser.add_argument("--end-frame", type=int,
                        help="Ending frame number for processing multiple frames")
    parser.add_argument("--loop", action="store_true",
                        help="Process frames in a continuous loop from start to end")
    parser.add_argument("--stats", action="store_true",
                        help="Calculate and display detailed timing statistics")
    
    # Configuration options
    parser.add_argument("--config", type=str, 
                        help="Path to configuration YAML file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    # Model options
    parser.add_argument("--model-precision", type=str, choices=["fp16", "fp32", "int8"], default="fp16",
                        help="Model precision (default: fp16)")
    
    # Performance options
    parser.add_argument("--batch-size", type=int, 
                        help="Maximum processing batch size (default from config)")
    parser.add_argument("--max-workers", type=int,
                        help="Maximum number of worker threads (default from config)")
    parser.add_argument("--frame-cache-size", type=int, default=500,
                        help="Number of frames to keep in memory cache (default: 500)")
    parser.add_argument("--preload-all", action="store_true", default=True,
                        help="Preload all frames at the beginning (default: True)")
    parser.add_argument("--no-preload-all", action="store_false", dest="preload_all",
                        help="Disable preloading all frames at the beginning")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output (default: output)")
    parser.add_argument("--stream-only", action="store_true",
                        help="Only stream results without saving to disk")
    parser.add_argument("--batch-writes", action="store_true",
                        help="Batch writes to reduce disk I/O (default: False)")
    parser.add_argument("--write-interval", type=float, default=5.0,
                        help="Interval in seconds between batch writes (default: 5)")
    
    return parser.parse_args()

def process_frame(frame_number: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single frame for player and ball detection.
    
    Args:
        frame_number: The frame number to process
        config: Configuration dictionary
        
    Returns:
        Results dictionary with detected positions and orientations
    """
    global logger, _ALL_PRELOADED_FRAMES
    # Start timing
    total_start_time = time.time()
    
    # Create wall time dictionary to track real elapsed time for each phase
    wall_time = {
        'preprocessing': 0.0,
        'detection': 0.0,
        'pose_detection': 0.0,
        'triangulation': 0.0,
        'postprocessing': 0.0,
        'total': 0.0
    }
    
    # PHASE 1: PREPROCESSING - START
    preprocessing_start = time.time()
    
    # Use cached camera transformation data
    camera_data = load_camera_transforms_cached()
    
    # Use preloaded frames if available, otherwise load them
    if _ALL_PRELOADED_FRAMES and frame_number in _ALL_PRELOADED_FRAMES:
        preloaded_frames = _ALL_PRELOADED_FRAMES[frame_number]
        logger.debug(f"Using preloaded frames for frame {frame_number}")
    else:
        logger.debug(f"Preloaded frames not available for frame {frame_number}, loading on demand")
        preloaded_frames = preload_camera_frames(frame_number, camera_data)
    
    # Filter and collect valid camera frames for batched processing
    valid_cameras = []
    # Prepare batches for player and ball detection
    player_batch_frames = []
    ball_batch_frames = []
    player_batch_info = []
    ball_batch_info = []
    
    for frame_info in camera_data.get('frames', []):
        file_path = frame_info.get('file_path', '')
        
        # Skip if no file path or if it's a global camera
        if not file_path or 'global_' in file_path:
            continue
            
        # Skip if no transform matrix
        transform_matrix = frame_info.get('transform_matrix', [])
        if not transform_matrix:
            continue
            
        # Skip if no preloaded frame
        if file_path not in preloaded_frames:
            continue
            
        # Get frame image
        frame_img = preloaded_frames[file_path]
        
        # Extract camera parameters
        camera_name = os.path.basename(file_path)
        position = extract_camera_position(transform_matrix)
        direction = extract_camera_direction(transform_matrix)
        
        # Extract intrinsic parameters
        intrinsics = {
            'w': frame_info.get('w'),
            'h': frame_info.get('h'),
            'fl_x': frame_info.get('fl_x'),
            'fl_y': frame_info.get('fl_y'),
            'cx': frame_info.get('cx'),
            'cy': frame_info.get('cy')
        }
        
        # Validate intrinsics
        if not validate_intrinsics(intrinsics, camera_name):
            logger.warning(f"Invalid intrinsics for camera {camera_name}, skipping")
            continue
            
        # Add to valid cameras list
        valid_cameras.append({
            'frame_img': frame_img,
            'file_path': file_path,
            'camera_name': camera_name,
            'position': position,
            'direction': direction, 
            'transform_matrix': transform_matrix,
            'intrinsics': intrinsics
        })
        
        # Add to batch for player detection
        player_batch_frames.append(frame_img)
        player_batch_info.append({
            'file_path': file_path,
            'camera_name': camera_name,
            'position': position,
            'direction': direction,
            'transform_matrix': transform_matrix,
            'intrinsics': intrinsics
        })
        
        # Add to batch for ball detection
        ball_batch_frames.append(frame_img)
        ball_batch_info.append({
            'file_path': file_path,
            'camera_name': camera_name,
            'position': position,
            'direction': direction,
            'transform_matrix': transform_matrix,
            'intrinsics': intrinsics
        })
    
    logger.info(f"Processing {len(valid_cameras)} valid camera frames")
    
    # Maximum batch size for efficient processing - exactly matching camera count
    MAX_BATCH_SIZE = 20  # Exactly 20 cameras, process all in one batch
    
    # PHASE 1: PREPROCESSING - END
    wall_time['preprocessing'] = time.time() - preprocessing_start
    logger.info(f"Preprocessing completed in {wall_time['preprocessing']:.3f}s")
    
    # PHASE 2: DETECTION - START
    detection_start = time.time()
    
    # Create all batch tasks
    player_batch_tasks = []
    ball_batch_tasks = []
    
    # Prepare player batch tasks
    for batch_idx in range(0, len(player_batch_frames), MAX_BATCH_SIZE):
        batch_end = min(batch_idx + MAX_BATCH_SIZE, len(player_batch_frames))
        batch_frames = player_batch_frames[batch_idx:batch_end]
        batch_info = player_batch_info[batch_idx:batch_end]
        
        player_batch_tasks.append({
            'frames': batch_frames,
            'info': batch_info,
            'batch_idx': batch_idx,
            'model_path': config['player_model_path'],
            'conf_threshold': config['player_conf_threshold'],
            'type': 'player',
        })
    
    # Prepare ball batch tasks
    for batch_idx in range(0, len(ball_batch_frames), MAX_BATCH_SIZE):
        batch_end = min(batch_idx + MAX_BATCH_SIZE, len(ball_batch_frames))
        batch_frames = ball_batch_frames[batch_idx:batch_end]
        batch_info = ball_batch_info[batch_idx:batch_end]
        
        ball_batch_tasks.append({
            'frames': batch_frames,
            'info': batch_info,
            'batch_idx': batch_idx,
            'model_path': config['ball_model_path'],
            'conf_threshold': config['ball_conf_threshold'],
            'type': 'ball',
        })
    
    # Use ThreadPoolExecutor to process both player and ball detection in parallel
    player_detections_by_camera = {}
    ball_detections_by_camera = {}
    
    # Define a function to process a batch
    def process_batch(task):
        try:
            batch_type = task['type']
            batch_idx = task['batch_idx']
            model_path = task['model_path']
            conf_threshold = task['conf_threshold']
            batch_frames = task['frames']
            batch_info = task['info']
            
            logger.info(f"Processing {batch_type} detection batch {batch_idx//MAX_BATCH_SIZE + 1} with {len(batch_frames)} frames")
            
            # Process batch with YOLO model directly
            batch_detections = perform_yolo_inference_batched(
                model_path,
                batch_frames,
                conf_threshold,
                config,
            )
            
            # Return results with info
            return {
                'type': batch_type,
                'detections': batch_detections,
                'info': batch_info
            }
        except Exception as e:
            logger.error(f"Error processing {batch_type} detection batch {batch_idx//MAX_BATCH_SIZE + 1}: {e}")
            return {
                'type': batch_type,
                'detections': [None] * len(batch_frames),
                'info': batch_info
            }
    
    # Process all batches in parallel
    all_tasks = player_batch_tasks + ball_batch_tasks
    
    # Calculate optimal number of workers for batch processing
    batch_workers = min(len(all_tasks), config['max_workers'])
    logger.info(f"Processing {len(all_tasks)} detection batches using {batch_workers} parallel workers")
    
    # Use a reusable thread pool for batch processing
    executor = get_thread_pool('batch_processing', max_workers=batch_workers)
    
    # Submit all tasks
    futures = {executor.submit(process_batch, task): task for task in all_tasks}
    
    # Process results as they complete
    for future in as_completed(futures):
        try:
            result = future.result()
            batch_type = result['type']
            batch_detections = result['detections']
            batch_info = result['info']
            
            # Store results by camera
            for i, (detection, info) in enumerate(zip(batch_detections, batch_info)):
                if detection is not None:
                    if batch_type == 'player':
                        player_detections_by_camera[info['file_path']] = detection
                    else:  # ball
                        ball_detections_by_camera[info['file_path']] = detection
        except Exception as e:
            task = futures[future]
            logger.error(f"Error processing {task['type']} batch {task['batch_idx']//MAX_BATCH_SIZE + 1}: {e}")
    
    logger.info(f"Completed all detection batches: {len(player_detections_by_camera)} player and {len(ball_detections_by_camera)} ball detections")
    
    # PHASE 2: DETECTION - END
    wall_time['detection'] = time.time() - detection_start
    logger.info(f"Detection completed in {wall_time['detection']:.3f}s")
    
    # PHASE 3: POSE ESTIMATION - START
    pose_start = time.time()
    
    # Process results for each camera (pose estimation and ray creation)
    all_player_rays = []
    all_ball_rays = []
    all_pose_rays = {i: [] for i in range(33)}  # MediaPipe has 33 keypoints
    
    pose_processing_tasks = []
    
    # Prepare pose processing tasks
    for camera in valid_cameras:
        file_path = camera['file_path']
        
        # Process player detections for this camera
        if file_path in player_detections_by_camera:
            player_detection = player_detections_by_camera[file_path]
            
            # Check if detection exists and has valid boxes
            if (player_detection is not None and 
                len(player_detection) > 0 and 
                hasattr(player_detection[0], 'xyxy') and 
                len(player_detection[0].xyxy) > 0):
                
                # Process each player detected in this camera view
                for player_idx, box in enumerate(player_detection[0].xyxy):
                    # Add to pose processing tasks
                    pose_processing_tasks.append({
                        'camera': camera,
                        'box': box,
                        'player_idx': player_idx
                    })
    
    # Define function to process a single player pose
    def process_player_pose(task):
        try:
            camera = task['camera']
            box = task['box']
            player_idx = task['player_idx']
            
            file_path = camera['file_path']
            frame_img = camera['frame_img']
            camera_name = camera['camera_name']
            position = camera['position']
            direction = camera['direction']
            transform_matrix = camera['transform_matrix']
            intrinsics = camera['intrinsics']
            
            # Convert to tuple format for caching
            intrinsics_tuple = (
                intrinsics['w'], intrinsics['h'], 
                intrinsics['fl_x'], intrinsics['fl_y'],
                intrinsics['cx'], intrinsics['cy']
            )
            position_tuple = tuple(position)
            direction_tuple = tuple(direction)
            transform_matrix_np = np.array(transform_matrix)
            transform_matrix_tuple = tuple(tuple(row) for row in transform_matrix_np)
            
            # Get player position ray
            bottom_center = np.array([
                (box[0] + box[2]) / 2,  # x center
                intrinsics['h'] - box[3]  # flip y coordinate - use bottom center
            ])
            bottom_center_tuple = tuple(bottom_center)
            
            # Create ray from camera to bottom center point
            player_ray = get_ray_from_camera(
                position_tuple, 
                direction_tuple, 
                bottom_center_tuple, 
                intrinsics_tuple, 
                transform_matrix_tuple
            )
            
            # Process pose for this player
            landmarks = process_pose_from_detection(
                frame_img, box, intrinsics, transform_matrix, 
                player_id=player_idx, camera_name=camera_name, config=config
            )
            
            # Initialize pose rays for this player
            pose_rays = {}
            
            if landmarks:
                # Create rays for each keypoint
                for i, (x, y, z, visibility) in enumerate(landmarks):
                    if visibility > config['pose_visibility_threshold']:
                        point_2d = np.array([x, intrinsics['h'] - y])  # Flip y coordinate
                        point_2d_tuple = tuple(point_2d)
                        
                        # Create ray from camera to keypoint
                        keypoint_ray = get_ray_from_camera(
                            position_tuple, 
                            direction_tuple, 
                            point_2d_tuple, 
                            intrinsics_tuple, 
                            transform_matrix_tuple
                        )
                        pose_rays[i] = keypoint_ray
            
            return {
                'player_ray': player_ray,
                'pose_rays': pose_rays,
                'camera_name': camera_name,
                'player_idx': player_idx
            }
        except Exception as e:
            logger.error(f"Error processing player pose in camera {camera['camera_name']}: {e}")
            return {
                'player_ray': None,
                'pose_rays': {},
                'camera_name': camera['camera_name'],
                'player_idx': player_idx
            }
    
    # Process results in parallel
    if pose_processing_tasks:
        pose_workers = min(len(pose_processing_tasks), config.get('mp_pose_pool_size', 4))
        logger.info(f"Processing {len(pose_processing_tasks)} player poses using {pose_workers} parallel workers")
        
        # Use a reusable thread pool for pose processing
        executor = get_thread_pool('pose_processing', max_workers=pose_workers)
        
        # Submit all tasks
        pose_futures = [executor.submit(process_player_pose, task) for task in pose_processing_tasks]
        
        # Process results as they complete
        for future in as_completed(pose_futures):
            try:
                result = future.result()
                if result['player_ray'] is not None:
                    all_player_rays.append(result['player_ray'])
                
                # Add pose rays to global collection
                for keypoint_id, ray in result['pose_rays'].items():
                    all_pose_rays[keypoint_id].append(ray)
                
            except Exception as e:
                logger.error(f"Error processing player pose result: {e}")
    
    # Process ball rays (simpler, can be done sequentially)
    for camera in valid_cameras:
        file_path = camera['file_path']
        
        # Process ball detections for this camera
        if file_path in ball_detections_by_camera:
            ball_detection = ball_detections_by_camera[file_path]
            
            # Check if detection exists and has valid confidence values
            if (ball_detection is not None and 
                len(ball_detection) > 0 and 
                hasattr(ball_detection[0], 'confidence') and 
                len(ball_detection[0].confidence) > 0):
                
                # Extract camera parameters
                position = camera['position']
                direction = camera['direction']
                transform_matrix = camera['transform_matrix']
                intrinsics = camera['intrinsics']
                
                # Convert to tuple format for caching
                intrinsics_tuple = (
                    intrinsics['w'], intrinsics['h'], 
                    intrinsics['fl_x'], intrinsics['fl_y'],
                    intrinsics['cx'], intrinsics['cy']
                )
                position_tuple = tuple(position)
                direction_tuple = tuple(direction)
                transform_matrix_np = np.array(transform_matrix)
                transform_matrix_tuple = tuple(tuple(row) for row in transform_matrix_np)
                
                # Get the detection with highest confidence
                best_detection_idx = np.argmax(ball_detection[0].confidence)
                box = ball_detection[0].xyxy[best_detection_idx]
                
                # Validate box coordinates
                if box[0] < box[2] and box[1] < box[3]:
                    # Get ball position ray
                    bottom_center = np.array([
                        (box[0] + box[2]) / 2,  # x center
                        intrinsics['h'] - box[3]  # flip y coordinate - use bottom center
                    ])
                    bottom_center_tuple = tuple(bottom_center)
                    
                    # Create ray from camera to ball center point
                    ray = get_ray_from_camera(
                        position_tuple, 
                        direction_tuple, 
                        bottom_center_tuple, 
                        intrinsics_tuple, 
                        transform_matrix_tuple
                    )
                    all_ball_rays.append(ray)
    
    logger.info(f"Collected {len(all_player_rays)} player rays and {len(all_ball_rays)} ball rays")
    
    # PHASE 3: POSE ESTIMATION - END
    wall_time['pose_detection'] = time.time() - pose_start
    logger.info(f"Pose detection completed in {wall_time['pose_detection']:.3f}s")
    
    # PHASE 4: TRIANGULATION - START
    triangulation_start = time.time()
    
    # Find player and ball positions using ray intersection and triangulate pose in parallel
    player_position = None
    ball_position = None
    pose_results = None
    
    # Define a function to find player position
    def find_player_position():
        if all_player_rays:
            return find_ray_intersection(all_player_rays, "player", config)
        return None
    
    # Define a function to find ball position
    def find_ball_position():
        if all_ball_rays:
            return find_ray_intersection(all_ball_rays, "ball", config)
        return None
    
    # Define a function to triangulate pose
    def process_pose():
        return triangulate_pose(all_pose_rays, config)
    
    # Run all triangulation tasks in parallel
    # Use a reusable thread pool for triangulation
    executor = get_thread_pool('triangulation', max_workers=3)
    
    # Submit all tasks
    player_future = executor.submit(find_player_position)
    ball_future = executor.submit(find_ball_position)
    pose_future = executor.submit(process_pose)
    
    # Get results
    player_position = player_future.result()
    ball_position = ball_future.result()
    pose_3d, orientation_vector, cardinal_direction = pose_future.result()
    
    # Log positions
    if player_position is not None:
        logger.info(f"Player position: [{player_position[0]:.2f}, {player_position[1]:.2f}, {player_position[2]:.2f}]")
    
    if ball_position is not None:
        logger.info(f"Ball position: [{ball_position[0]:.2f}, {ball_position[1]:.2f}, {ball_position[2]:.2f}]")
    
    # PHASE 4: TRIANGULATION - END
    wall_time['triangulation'] = time.time() - triangulation_start
    logger.info(f"Triangulation completed in {wall_time['triangulation']:.3f}s")
    
    # PHASE 5: POSTPROCESSING - START
    postprocessing_start = time.time()
    
    # Create results dictionary
    results = create_output(frame_number, player_position, ball_position, pose_3d, orientation_vector, cardinal_direction)
    
    # PHASE 5: POSTPROCESSING - END
    wall_time['postprocessing'] = time.time() - postprocessing_start
    logger.info(f"Postprocessing completed in {wall_time['postprocessing']:.3f}s")
    
    # Calculate total wall time
    wall_time['total'] = time.time() - total_start_time
    logger.info(f"Total processing time: {wall_time['total']:.3f}s")
    
    # Store wall times for stats reporting
    # Import global TIMING dict to store our wall times
    from soccer3d.models.detection import TIMING as detection_timing
    detection_timing['wall_time'] = wall_time
    
    return results


def create_output(
    frame_number: int,
    player_position: Optional[np.ndarray],
    ball_position: Optional[np.ndarray],
    pose_3d: Dict[int, np.ndarray],
    orientation_vector: Optional[np.ndarray],
    cardinal_direction: Optional[str]
) -> Dict[str, Any]:
    """
    Create a structured output dictionary with player and ball information.
    
    Args:
        frame_number: Frame number being processed
        player_position: 3D position of player
        ball_position: 3D position of ball
        pose_3d: Dictionary of pose keypoints
        orientation_vector: Player orientation vector
        cardinal_direction: Text description of player orientation
        
    Returns:
        Dictionary ready to be serialized to JSON
    """
    # Import POSE_KEYPOINT_NAMES for creating keypoint names
    from soccer3d.models.pose import POSE_KEYPOINT_NAMES
    
    # Create base structure with field orientation information
    output = {
        "frame_info": {
            "frame_number": frame_number,
            "timestamp": datetime.now().isoformat()
        },
        "field_orientation": {
            "x_axis": "side to side (width of field)",
            "y_axis": "goal to goal (length of field)",
            "z_axis": "up (height)",
            "origin": "center of field",
            "units": "meters"
        },
        "player": {
            "detected": player_position is not None
        },
        "ball": {
            "detected": ball_position is not None
        }
    }
    
    # Add player information if detected
    if player_position is not None:
        output["player"]["position"] = {
            "x": float(player_position[0]),
            "y": float(player_position[1]),
            "z": float(player_position[2])
        }
        
        # Add orientation if available
        if orientation_vector is not None:
            output["player"]["orientation"] = {
                "vector": {
                    "x": float(orientation_vector[0]),
                    "y": float(orientation_vector[1]),
                    "z": float(orientation_vector[2])
                },
                "description": cardinal_direction
            }
        
        # Add 3D pose keypoints
        if pose_3d:
            output["player"]["pose_keypoints"] = {}
            for keypoint_idx, position in pose_3d.items():
                keypoint_name = POSE_KEYPOINT_NAMES.get(keypoint_idx, f"unknown_{keypoint_idx}")
                output["player"]["pose_keypoints"][keypoint_name] = {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2])
                }
    
    # Add ball information if detected
    if ball_position is not None:
        output["ball"]["position"] = {
            "x": float(ball_position[0]),
            "y": float(ball_position[1]),
            "z": float(ball_position[2])
        }
    
    return output


def save_output(output_data: Dict[str, Any], output_dir: str, frame_number: int, stream_only: bool = False) -> str:
    """
    Save output data to a JSON file and publish to MQTT broker.
    Uses batching to reduce disk I/O.
    
    Args:
        output_data: Output data dictionary
        output_dir: Directory to save output
        frame_number: Frame number for filename
        stream_only: Whether to only stream results without saving to disk
        
    Returns:
        Path to saved file or empty string if skipped
    """
    global _OUTPUT_CACHE, _OUTPUT_CACHE_LOCK, _LAST_WRITE_TIME
    
    # Convert output data to JSON string (needed for streaming anyway)
    json_str = json.dumps(output_data, indent=2)
    
    # Stream results if streaming is enabled
    # This is where you would add code to stream via MQTT, WebSockets, etc.
    # For example, if using MQTT:
    #
    # if mqtt_client and mqtt_topic:
    #     mqtt_client.publish(mqtt_topic, json_str)
    
    # If we're only streaming, return early
    if stream_only:
        return ""
    
    filepath = ""
    
    # Get current time for batching decision
    current_time = time.time()
    
    with _OUTPUT_CACHE_LOCK:
        # Add to output cache
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"soccer3d_frame_{frame_number}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        _OUTPUT_CACHE.append({
            'json_str': json_str,
            'filepath': filepath,
            'frame_number': frame_number
        })
        
        # Check if it's time to write the batch or if this is the first write
        should_write = (current_time - _LAST_WRITE_TIME) >= _WRITE_INTERVAL or _LAST_WRITE_TIME == 0
        
        # If it's time to write or we have too many cached outputs, write them to disk
        if should_write or len(_OUTPUT_CACHE) > 20:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Write all cached outputs
            frames_written = []
            for output in _OUTPUT_CACHE:
                with open(output['filepath'], 'w') as f:
                    f.write(output['json_str'])
                frames_written.append(output['frame_number'])
            
            # Log what we wrote
            logger.info(f"Batch wrote {len(frames_written)} frames to {output_dir} (frames {min(frames_written)}-{max(frames_written)})")
            
            # Clear the cache and update the last write time
            _OUTPUT_CACHE.clear()
            _LAST_WRITE_TIME = current_time
        else:
            # If we're not writing yet, log that we've cached the output
            logger.debug(f"Cached output for frame {frame_number} (will write in batch later)")
    
    return filepath


def collect_timing_stats() -> Dict[str, Dict[str, float]]:
    """
    Collect timing statistics from all modules.
    
    Returns:
        Dictionary containing wall time statistics for each pipeline phase
    """
    # Import TIMING dictionaries from all modules to get wall time data
    from soccer3d.models.detection import TIMING as detection_timing
    
    # Get the wall time measurements
    wall_time = detection_timing.get('wall_time', {})
    
    if not wall_time:
        # Fallback if no wall time recorded
        return {
            "error": {
                "message": "No wall time measurements available"
            }
        }
    
    # Collect all stats into categories using real wall time
    stats = {
        "preprocessing": {
            "wall_time": wall_time.get('preprocessing', 0.0),
        },
        "detection": {
            "wall_time": wall_time.get('detection', 0.0),
        },
        "pose_detection": {
            "wall_time": wall_time.get('pose_detection', 0.0),
        },
        "triangulation": {
            "wall_time": wall_time.get('triangulation', 0.0),
        },
        "postprocessing": {
            "wall_time": wall_time.get('postprocessing', 0.0),
        },
        "total": {
            "wall_time": wall_time.get('total', 0.0),
        }
    }
    
    # Calculate percentage of total for each phase
    total_wall_time = wall_time.get('total', 0.0)
    if total_wall_time > 0:
        for category, timings in stats.items():
            if category != "total":
                phase_time = timings.get("wall_time", 0.0)
                timings["percentage"] = (phase_time / total_wall_time) * 100
    
    return stats


def print_timing_stats(stats: Dict[str, Dict[str, float]], total_frames: int = 1) -> None:
    """
    Print timing statistics in a formatted way.
    
    Args:
        stats: Dictionary of timing statistics
        total_frames: Number of frames processed
    """
    if "error" in stats:
        logger.error(f"Error in timing stats: {stats['error']['message']}")
        return
    
    # Print header
    logger.info("=" * 80)
    logger.info(f"SOCCER3D WALL TIME STATISTICS (averaged over {total_frames} frames)")
    logger.info("=" * 80)
    
    # Get total time
    total_time = stats.get("total", {}).get("wall_time", 0.0)
    per_frame_total = total_time / total_frames
    
    # Print each phase in pipeline order
    phases = ["preprocessing", "detection", "pose_detection", "triangulation", "postprocessing"]
    
    for phase in phases:
        phase_stats = stats.get(phase, {})
        wall_time = phase_stats.get("wall_time", 0.0)
        percentage = phase_stats.get("percentage", 0.0)
        per_frame = wall_time / total_frames
        
        logger.info(f"{phase.upper()}: {wall_time:.3f}s total, {per_frame:.3f}s/frame ({percentage:.1f}% of pipeline)")
    
    # Print the grand total
    logger.info("-" * 80)
    logger.info(f"TOTAL PIPELINE: {total_time:.3f}s total, {per_frame_total:.3f}s/frame")
    
    # Calculate internal FPS (pipeline-only)
    internal_fps = total_frames / total_time if total_time > 0 else 0
    logger.info(f"Internal processing speed: {internal_fps:.2f} FPS (pipeline only)")
    
    # Print real-world FPS if available
    real_wall_time = stats.get("total", {}).get("real_wall_time", 0.0)
    real_fps = stats.get("total", {}).get("real_fps", 0.0)
    
    if real_wall_time > 0:
        logger.info("-" * 80)
        logger.info(f"REAL WALL CLOCK TIME: {real_wall_time:.2f}s total, {real_wall_time/total_frames:.3f}s/frame")
        logger.info(f"ACTUAL PROCESSING SPEED: {real_fps:.2f} FPS (including I/O, saving, etc.)")
    
    logger.info("=" * 80)


def main():
    """Main entry point for the Soccer3D command-line tool."""
    global logger, config, _FRAME_CACHE_MAX_SIZE, _WRITE_INTERVAL, _ALL_PRELOADED_FRAMES
    
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize Soccer3D library - explicitly assign to our global variables
    config, logger = initialize(args.config)
    
    # Override configuration with command-line arguments
    if args.log_level:
        config['log_level'] = args.log_level
    if args.max_workers:
        config['max_workers'] = args.max_workers
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Update frame cache size if specified
    if args.frame_cache_size:
        from soccer3d.utils.camera import _FRAME_CACHE_MAX_SIZE
        _FRAME_CACHE_MAX_SIZE = args.frame_cache_size
        logger.info(f"Setting frame cache size to {_FRAME_CACHE_MAX_SIZE}")
    
    # Update write interval if batch writes enabled
    if args.batch_writes:
        _WRITE_INTERVAL = args.write_interval
        logger.info(f"Batch writes enabled with interval of {_WRITE_INTERVAL}s")
    
    # Add model precision configuration
    config['model_precision'] = args.model_precision
    logger.info(f"Using model precision: {config['model_precision']}")
    
    # Show current configuration
    logger.info(f"Using configuration: {config}")
    
    # Initialize models
    with SuppressOutput():
        # Initialize MediaPipe pose model pool
        logger.info("Initializing MediaPipe pose model pool...")
        initialize_mp_pose_pool(config)
        
        # Warm up PyTorch CUDA
        warmup_pytorch_cuda()
        
        # Warm up YOLO models
        warmup_yolo_models(config)
    
    # Handle stream-only mode
    if args.stream_only:
        logger.info("Stream-only mode enabled - no output files will be written")
    
    # Determine frame range to process
    start_frame = args.start_frame if args.start_frame is not None else args.frame
    end_frame = args.end_frame if args.end_frame is not None else start_frame
    
    # Make sure start_frame <= end_frame
    if start_frame > end_frame:
        logger.warning(f"Start frame ({start_frame}) is greater than end frame ({end_frame}). Swapping values.")
        start_frame, end_frame = end_frame, start_frame
    
    logger.info(f"Frame processing range: {start_frame} to {end_frame}")
    if args.loop:
        logger.info("Loop mode enabled: Will process frames continuously until interrupted")
        if not args.stream_only:
            logger.warning("Loop mode is enabled but --stream-only is not. This may generate a lot of files.")
    
    # Preload all frames at once if processing multiple frames and preload_all is enabled
    if args.preload_all and end_frame > start_frame:
        logger.info(f"Preloading all frames from {start_frame} to {end_frame}...")
        from soccer3d.utils.camera import preload_all_frames
        _ALL_PRELOADED_FRAMES = preload_all_frames(start_frame, end_frame)
        logger.info(f"Preloaded {len(_ALL_PRELOADED_FRAMES)} frames ({sum(len(frames) for frames in _ALL_PRELOADED_FRAMES.values())} total images)")
    else:
        if not args.preload_all:
            logger.info("Frame preloading disabled, will load frames on demand")
        elif end_frame == start_frame:
            logger.info("Processing single frame, no need to preload multiple frames")
    
    # Start wall clock timing for entire processing
    processing_start_time = time.time()
    
    try:
        # Process frames
        current_frame = start_frame
        frames_processed = 0
        
        while True:
            logger.info(f"Processing frame {current_frame}")
            
            try:
                # Process the frame
                results = process_frame(current_frame, config)
                frames_processed += 1
                
                # Save the results
                if results:
                    save_output(results, args.output_dir, current_frame, args.stream_only)
                else:
                    logger.error(f"No results generated for frame {current_frame}")
            except Exception as e:
                logger.error(f"Error processing frame {current_frame}: {e}")
            
            # Move to next frame
            current_frame += 1
            
            # Check if we've reached the end of the range
            if current_frame > end_frame:
                if args.loop:
                    logger.info(f"Reached end frame {end_frame}, looping back to start frame {start_frame}")
                    current_frame = start_frame
                    # Add a small delay to avoid overwhelming the system
                    time.sleep(0.1)
                else:
                    logger.info(f"Reached end frame {end_frame}, processing complete")
                    break
                    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Exiting gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Make sure to clean up thread pools
        shutdown_thread_pools()
    
    # End wall clock timing
    processing_end_time = time.time()
    total_wall_time = processing_end_time - processing_start_time
    real_fps = frames_processed / total_wall_time if total_wall_time > 0 else 0
    
    # Print timing statistics if requested
    if args.stats and frames_processed > 0:
        try:
            logger.info(f"Collecting timing statistics for {frames_processed} processed frames...")
            stats = collect_timing_stats()
            
            # Add real wall time to stats
            if "total" not in stats:
                stats["total"] = {}
            stats["total"]["real_wall_time"] = total_wall_time
            stats["total"]["real_fps"] = real_fps
            
            print_timing_stats(stats, frames_processed)
            
            # Log actual wall clock time separately
            logger.info("=" * 80)
            logger.info(f"REAL WALL CLOCK STATISTICS:")
            logger.info(f"Total wall clock time: {total_wall_time:.2f}s for {frames_processed} frames")
            logger.info(f"Real processing speed: {real_fps:.2f} FPS (frames per second)")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Error collecting timing statistics: {e}")
    
    logger.info("Soccer3D processing complete")


if __name__ == "__main__":
    main()