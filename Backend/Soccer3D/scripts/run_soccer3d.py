#!/usr/bin/env python
"""
Soccer3D Command Line Tool

This script provides a command-line interface to the Soccer3D library.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Add parent directory to path to allow running script from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import MQTT client
try:
    import paho.mqtt.client as mqtt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paho-mqtt"])
    import paho.mqtt.client as mqtt

# Import Soccer3D modules
from soccer3d import initialize, CONFIG, logger
from soccer3d.models import (
    initialize_triton_clients,
    check_models_ready,
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
    
    # MQTT options
    parser.add_argument("--mqtt-broker", type=str, default="localhost",
                        help="MQTT broker address (default: localhost)")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port (default: 1883)")
    parser.add_argument("--mqtt-topic", type=str, default="soccer3d/data",
                        help="MQTT topic for publishing data (default: soccer3d/data)")
    parser.add_argument("--mqtt-disable", action="store_true",
                        help="Disable MQTT publishing")
    
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
    # Start timing
    total_start_time = time.time()
    
    # Load camera transformation data
    try:
        with open('dataset/per_cam_transforms.json', 'r') as f:
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
    
    # STEP 2: Process camera frames in parallel
    all_player_rays = []
    all_ball_rays = []
    all_pose_rays = {i: [] for i in range(33)}  # MediaPipe has 33 keypoints
    
    # Process each camera (broadcast cameras only, not global views)
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
        
        # Detect players
        player_detections = perform_yolo_inference_batched(
            config['player_model_url'], 
            [frame_img], 
            config['player_conf_threshold'],
            config
        )
        
        # Detect ball
        ball_detections = perform_yolo_inference_batched(
            config['ball_model_url'], 
            [frame_img], 
            config['ball_conf_threshold'],
            config
        )
        
        # Process player detections
        if player_detections and len(player_detections) > 0 and player_detections[0]:
            for idx, box in enumerate(player_detections[0].xyxy):
                # Validate box coordinates
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                
                # Get player position ray
                bottom_center = np.array([
                    (box[0] + box[2]) / 2,  # x center
                    intrinsics['h'] - box[3]  # flip y coordinate - use bottom center
                ])
                bottom_center_tuple = tuple(bottom_center)
                
                # Create ray from camera to bottom center point
                ray = get_ray_from_camera(
                    position_tuple, 
                    direction_tuple, 
                    bottom_center_tuple, 
                    intrinsics_tuple, 
                    transform_matrix_tuple
                )
                all_player_rays.append(ray)
                
                # Process pose for this player
                landmarks = process_pose_from_detection(
                    frame_img, box, intrinsics, transform_matrix, 
                    player_id=idx, camera_name=camera_name, config=config
                )
                
                if landmarks:
                    # Create rays for each keypoint
                    for i, (x, y, z, visibility) in enumerate(landmarks):
                        if visibility > config['pose_visibility_threshold']:
                            point_2d = np.array([x, intrinsics['h'] - y])  # Flip y coordinate
                            point_2d_tuple = tuple(point_2d)
                            
                            # Create ray from camera to keypoint
                            ray = get_ray_from_camera(
                                position_tuple, 
                                direction_tuple, 
                                point_2d_tuple, 
                                intrinsics_tuple, 
                                transform_matrix_tuple
                            )
                            all_pose_rays[i].append(ray)
        
        # Process ball detections - only use the highest confidence detection
        if ball_detections and len(ball_detections) > 0 and ball_detections[0]:
            if len(ball_detections[0].confidence) > 0:
                # Get the detection with highest confidence
                best_detection_idx = np.argmax(ball_detections[0].confidence)
                box = ball_detections[0].xyxy[best_detection_idx]
                
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
    
    # STEP 3: Find player and ball positions using ray intersection
    player_position = None
    if all_player_rays:
        player_position = find_ray_intersection(all_player_rays, "player", config)
        if player_position is not None:
            logger.info(f"Player position: [{player_position[0]:.2f}, {player_position[1]:.2f}, {player_position[2]:.2f}]")
    
    ball_position = None
    if all_ball_rays:
        ball_position = find_ray_intersection(all_ball_rays, "ball", config)
        if ball_position is not None:
            logger.info(f"Ball position: [{ball_position[0]:.2f}, {ball_position[1]:.2f}, {ball_position[2]:.2f}]")
    
    # STEP 4: Triangulate player pose
    pose_3d, orientation_vector, cardinal_direction = triangulate_pose(all_pose_rays, config)
    
    # STEP 5: Create results dictionary
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


def save_output(output_data: Dict[str, Any], output_dir: str, frame_number: int, 
                mqtt_config: Dict[str, Any] = None) -> str:
    """
    Save output data to a JSON file and publish to MQTT broker.
    
    Args:
        output_data: Output data dictionary
        output_dir: Directory to save output
        frame_number: Frame number for filename
        mqtt_config: MQTT configuration dictionary (broker, port, topic)
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"soccer3d_frame_{frame_number}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert output data to JSON string
    json_str = json.dumps(output_data, indent=2)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(json_str)
    
    logger.info(f"Output saved to {filepath}")
    
    # Publish to MQTT broker if enabled
    if mqtt_config and not mqtt_config.get('disable', False):
        try:
            broker = mqtt_config.get('broker', 'localhost')
            port = mqtt_config.get('port', 1883)
            topic = mqtt_config.get('topic', 'soccer3d/data')
            
            # Create MQTT client
            client = mqtt.Client()
            
            # Connect to broker
            logger.info(f"Connecting to MQTT broker at {broker}:{port}")
            client.connect(broker, port, 60)
            
            # Publish data
            logger.info(f"Publishing data to topic: {topic}")
            result = client.publish(topic, json_str)
            
            # Check if publish was successful
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("Data successfully published to MQTT broker")
            else:
                logger.warning(f"Failed to publish data to MQTT broker: {result.rc}")
            
            # Disconnect
            client.disconnect()
        except Exception as e:
            logger.error(f"Error publishing to MQTT broker: {e}")
    
    return filepath


def main():
    """Main entry point for the Soccer3D command-line tool."""
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize Soccer3D library
    config, _ = initialize(args.config)
    
    # Override configuration with command-line arguments
    if args.player_model_url:
        config['player_model_url'] = args.player_model_url
    if args.ball_model_url:
        config['ball_model_url'] = args.ball_model_url
    if args.log_level:
        config['log_level'] = args.log_level
    
    # Create MQTT configuration
    mqtt_config = {
        'broker': args.mqtt_broker,
        'port': args.mqtt_port,
        'topic': args.mqtt_topic,
        'disable': args.mqtt_disable
    }
    
    # Show current configuration
    logger.info(f"Using configuration: {config}")
    logger.info(f"MQTT configuration: {mqtt_config}")
    
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
            save_output(results, args.output_dir, frame_number, mqtt_config)
        else:
            logger.error(f"No results generated for frame {frame_number}")
    except Exception as e:
        logger.error(f"Error processing frame {frame_number}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
