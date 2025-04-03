"""
Camera utility functions for Soccer3D.

This module handles camera transformation and intrinsics.
"""
import numpy as np
import logging
from typing import Dict, List, Any, Union

logger = logging.getLogger("Soccer3D")


def extract_camera_position(transform_matrix: List[List[float]]) -> np.ndarray:
    """
    Extract camera position from the transformation matrix.
    
    Args:
        transform_matrix: 4x4 camera transformation matrix
        
    Returns:
        3D position vector as numpy array
    """
    try:
        return np.array([transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]])
    except (IndexError, TypeError) as e:
        logger.error(f"Invalid transform matrix format: {e}")
        return np.array([0.0, 0.0, 0.0])  # Return safe default value


def extract_camera_direction(transform_matrix: List[List[float]]) -> np.ndarray:
    """
    Extract camera viewing direction from the transformation matrix.
    
    Args:
        transform_matrix: 4x4 camera transformation matrix
        
    Returns:
        3D direction vector as numpy array
    """
    try:
        rotation = np.array([
            transform_matrix[0][:3],
            transform_matrix[1][:3],
            transform_matrix[2][:3]
        ])
        
        # Get the camera's forward direction (negative z-axis in camera space)
        forward = rotation @ np.array([0, 0, -1])
        return forward
    except (IndexError, TypeError, ValueError) as e:
        logger.error(f"Invalid transform matrix format for direction extraction: {e}")
        return np.array([0.0, 0.0, -1.0])  # Return safe default value


def validate_intrinsics(intrinsics: Dict[str, Any], camera_name: str = "unknown") -> bool:
    """
    Validate camera intrinsic parameters.
    
    Args:
        intrinsics: Camera intrinsic parameters
        camera_name: Name of the camera for logging
        
    Returns:
        True if intrinsics are valid, False otherwise
    """
    required_keys = ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']
    
    # Check that all required keys exist and have non-zero values
    for key in required_keys:
        if key not in intrinsics or not intrinsics[key]:
            logger.warning(f"Missing or invalid intrinsic parameter '{key}' for camera {camera_name}")
            return False
    
    # Check for reasonable focal length values
    if intrinsics['fl_x'] <= 0 or intrinsics['fl_y'] <= 0:
        logger.warning(f"Invalid focal length for camera {camera_name}: fl_x={intrinsics['fl_x']}, fl_y={intrinsics['fl_y']}")
        return False
    
    # Make sure principal point is within image
    if (intrinsics['cx'] < 0 or intrinsics['cx'] >= intrinsics['w'] or 
        intrinsics['cy'] < 0 or intrinsics['cy'] >= intrinsics['h']):
        logger.warning(f"Principal point outside image for camera {camera_name}")
        return False
    
    return True


def convert_transform_matrix(transform_matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Convert a transform matrix to numpy array, performing validation.
    
    Args:
        transform_matrix: 4x4 camera transformation matrix
        
    Returns:
        Validated transform matrix as numpy array
    """
    try:
        # Convert to numpy array if not already
        if isinstance(transform_matrix, list):
            transform_matrix = np.array(transform_matrix)
        
        # Check shape
        if transform_matrix.shape != (4, 4):
            logger.warning(f"Transform matrix has incorrect shape: {transform_matrix.shape}, expected (4, 4)")
            # Try to fix common issue with 3x4 matrices
            if transform_matrix.shape == (3, 4):
                # Add the missing row [0, 0, 0, 1]
                transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])
                logger.info("Fixed 3x4 transform matrix by adding the missing row [0, 0, 0, 1]")
            else:
                # Return identity matrix as fallback
                logger.error("Cannot fix transform matrix shape, using identity matrix")
                return np.eye(4)
        
        return transform_matrix
    except Exception as e:
        logger.error(f"Error converting transform matrix: {e}")
        return np.eye(4)  # Return identity matrix as fallback


def preload_camera_frames(frame_number: int, camera_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Preload camera frames for a given frame number.
    
    Args:
        frame_number: Frame number to preload
        camera_data: Camera data including file paths
        
    Returns:
        Dictionary mapping camera file paths to frames
    """
    import os
    import cv2
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    logger.info(f"Preloading camera frames for frame {frame_number}")
    
    # Create a thread-safe dictionary
    preloaded_frames = {}
    preloaded_frames_lock = threading.RLock()
    
    # Get all frames from camera data
    all_frames = camera_data.get('frames', [])
    valid_camera_frames = []
    
    # First identify all valid frames to load
    for frame_info in all_frames:
        file_path = frame_info.get('file_path', '')
        
        # Skip if no file path or if it's a global camera (handled separately)
        if not file_path or 'global_' in file_path:
            continue
            
        # Prepare image path
        image_path = os.path.join('output_frames', file_path, f"frame_{frame_number}.jpg")
        if os.path.exists(image_path):
            valid_camera_frames.append((file_path, image_path))
    
    logger.info(f"Found {len(valid_camera_frames)} valid camera frames to load")
    
    # Define function to load a single image
    def load_single_image(args):
        file_path, image_path = args
        camera_name = os.path.basename(file_path)
        
        try:
            frame_img = cv2.imread(image_path)
            if frame_img is not None and frame_img.size > 0:
                with preloaded_frames_lock:
                    preloaded_frames[file_path] = frame_img
                return True, file_path
            else:
                logger.warning(f"Failed to read image for frame {frame_number} from camera {camera_name}")
                return False, file_path
        except Exception as e:
            logger.error(f"Error loading frame image from {camera_name}: {e}")
            return False, file_path
    
    # Load images in parallel
    start_time = __import__('time').time()
    load_success_count = 0
    
    # Determine max workers from config or use a default
    max_workers = 8  # Default
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(load_single_image, args): args[0]
                          for args in valid_camera_frames}
        
        for future in as_completed(future_to_path):
            try:
                success, file_path = future.result()
                if success:
                    load_success_count += 1
            except Exception as e:
                logger.error(f"Unhandled exception in parallel image loading: {e}")
    
    load_time = __import__('time').time() - start_time
    logger.info(f"Preloaded {load_success_count}/{len(valid_camera_frames)} camera frames in {load_time:.2f}s")
    
    return preloaded_frames
