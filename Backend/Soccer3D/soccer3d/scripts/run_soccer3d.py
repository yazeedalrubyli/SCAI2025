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
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np

# Local imports
from soccer3d import initialize
from soccer3d.models import (
    initialize_triton_clients,
    perform_yolo_inference_batched,
    warmup_triton_inference,
    initialize_mp_pose_pool,
    process_pose_from_detection,
    warmup_pytorch_cuda,
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
from soccer3d.utils.camera import preload_camera_frames
from soccer3d.logger import SuppressOutput

# Global variables
logger = None
config = None

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Soccer3D: Soccer player and ball 3D tracking")
    
    # Required/key options
    parser.add_argument("--frame", type=int, default=160,
                        help="Frame number to process (default: 160)")
    
    # Configuration options
    parser.add_argument("--config", type=str, 
                        help="Path to configuration YAML file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    # Triton server options
    parser.add_argument("--player-model-url", type=str,
                        help="URL for player detector model")
    parser.add_argument("--ball-model-url", type=str,
                        help="URL for ball detector model")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output (default: output)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional outputs")
    
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
    global logger
    # Start timing
    total_start_time = time.time()
    
    # Load camera transformation data
    try:
        with open('output_frames/per_cam_transforms.json', 'r') as f:
            camera_data = json.load(f)
        logger.info(f"Loaded camera transformation data with {len(camera_data.get('frames', []))} total cameras")
    except FileNotFoundError:
        logger.error("Error: Could not find the per_cam_transforms.json file.")
        return {}
    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON in per_cam_transforms.json file.")
        return {}
    
    # STEP 1: Preload all frames from output_frames directory into RAM 
    preload_start = time.time()
    preloaded_frames = preload_camera_frames(frame_number, camera_data)
    preload_time = time.time() - preload_start
    logger.info(f"Preloaded {len(preloaded_frames)} frames in {preload_time:.2f} seconds")
    
    # STEP 2: Prepare batched processing
    all_player_rays = []
    all_ball_rays = []
    all_pose_rays = {i: [] for i in range(33)}  # MediaPipe has 33 keypoints
    
    # Maximum batch size for efficient processing
    MAX_BATCH_SIZE = 20  # Match Triton server configuration
    
    # Prepare batches for player and ball detection
    player_batch_frames = []
    ball_batch_frames = []
    player_batch_info = []
    ball_batch_info = []
    
    # Filter and collect valid camera frames for batched processing
    valid_cameras = []
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
    
    logger.info(f"Processing {len(valid_cameras)} valid camera frames in batches of up to {MAX_BATCH_SIZE}")
    
    # STEP 3: Process player detection in batches
    player_detections_by_camera = {}
    ball_detections_by_camera = {}
    
    # Use ThreadPoolExecutor to process both player and ball detection in parallel
    
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
            'model_url': config['player_model_url'],
            'conf_threshold': config['player_conf_threshold'],
            'type': 'player'
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
            'model_url': config['ball_model_url'],
            'conf_threshold': config['ball_conf_threshold'],
            'type': 'ball'
        })
    
    # Define a function to process a batch
    def process_batch(task):
        try:
            batch_type = task['type']
            batch_idx = task['batch_idx']
            model_url = task['model_url']
            conf_threshold = task['conf_threshold']
            batch_frames = task['frames']
            batch_info = task['info']
            
            logger.info(f"Processing {batch_type} detection batch {batch_idx//MAX_BATCH_SIZE + 1} with {len(batch_frames)} frames")
            
            # Process batch
            batch_detections = perform_yolo_inference_batched(
                model_url,
                batch_frames,
                conf_threshold,
                config
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
    
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
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
    
    # STEP 5: Process results for each camera (pose estimation and ray creation)
    pose_processing_tasks = []
    
    for camera in valid_cameras:
        file_path = camera['file_path']
        # Skip if no player detection for this camera
        if file_path not in player_detections_by_camera:
            continue
            
        player_detection = player_detections_by_camera[file_path]
        # Check if detection exists and has valid boxes
        if (player_detection is None or
            len(player_detection) == 0 or
            not hasattr(player_detection[0], 'xyxy') or
            len(player_detection[0].xyxy) == 0):
            continue
            
        # Add pose processing tasks for each player in each camera
        for idx, box in enumerate(player_detection[0].xyxy):
            # Validate box coordinates
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
                
            pose_processing_tasks.append({
                'camera': camera,
                'box': box,
                'player_idx': idx
            })
    
    # Process player pose in parallel
    logger.info(f"Processing {len(pose_processing_tasks)} player poses in parallel")
    
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
    
    # Process all player poses in parallel
    pose_workers = min(len(pose_processing_tasks), config['max_workers'])
    if pose_processing_tasks:
        logger.info(f"Processing {len(pose_processing_tasks)} player poses using {pose_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=pose_workers) as executor:
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
    
    # STEP 6: Find player and ball positions using ray intersection and triangulate pose in parallel
    triangulation_start_time = time.time()  # Track triangulation time
    
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
    with ThreadPoolExecutor(max_workers=3) as executor:
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
        
    triangulation_time = time.time() - triangulation_start_time
    logger.info(f"Parallel triangulation completed in {triangulation_time:.3f}s")
    
    # STEP 7: Create results dictionary
    results = create_output(frame_number, player_position, ball_position, pose_3d, orientation_vector, cardinal_direction)
    
    # Log total processing time
    total_time = time.time() - total_start_time
    logger.info(f"Total processing time: {total_time:.2f}s")
    
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


def save_output(output_data: Dict[str, Any], output_dir: str, frame_number: int) -> str:
    """
    Save output data to a JSON file.
    
    Args:
        output_data: Output data dictionary
        output_dir: Directory to save output
        frame_number: Frame number for filename
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"soccer3d_frame_{frame_number}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Output saved to {filepath}")
    return filepath


def main():
    """Main entry point for the Soccer3D command-line tool."""
    global logger, config
    
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize Soccer3D library - explicitly assign to our global variables
    config, logger = initialize(args.config)
    
    # Override configuration with command-line arguments
    if args.player_model_url:
        config['player_model_url'] = args.player_model_url
    if args.ball_model_url:
        config['ball_model_url'] = args.ball_model_url
    if args.log_level:
        config['log_level'] = args.log_level
    
    # Show current configuration
    logger.info(f"Using configuration: {config}")
    
    # Initialize models
    with SuppressOutput():
        # Initialize MediaPipe pose model pool
        logger.info("Initializing MediaPipe pose model pool...")
        initialize_mp_pose_pool(config)
        
        # Initialize Triton clients
        logger.info("Initializing Triton clients...")
        if not initialize_triton_clients(config):
            logger.error("Failed to initialize Triton clients. Please ensure Triton server is running.")
            sys.exit(1)
        
        # Warm up models
        warmup_pytorch_cuda()
        warmup_triton_inference(config)
    
    # Process the requested frame
    frame_number = args.frame
    logger.info(f"Processing frame {frame_number}")
    
    try:
        # Process the frame
        results = process_frame(frame_number, config)
        
        # Save the results
        if results:
            save_output(results, args.output_dir, frame_number)
        else:
            logger.error(f"No results generated for frame {frame_number}")
    except Exception as e:
        logger.error(f"Error processing frame {frame_number}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
