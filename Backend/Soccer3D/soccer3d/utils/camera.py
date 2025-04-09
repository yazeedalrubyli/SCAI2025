"""
Camera utility functions for Soccer3D.

This module handles camera transformation and intrinsics.
"""
import numpy as np
import logging
from typing import Dict, List, Any, Union
import threading
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from soccer3d.utils.threading import get_thread_pool

logger = logging.getLogger("Soccer3D")

# Global camera transformation cache
_CAMERA_TRANSFORM_CACHE = None
_CAMERA_TRANSFORM_LOCK = threading.RLock()

# Global frame cache (LRU-like cache for frames)
_FRAME_CACHE = {}
_FRAME_CACHE_MAX_SIZE = 500  # Adjust based on available memory
_FRAME_CACHE_LOCK = threading.RLock()
_FRAME_CACHE_ACCESS_TIMES = {}  # Track last access for LRU eviction

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


def load_camera_transforms_cached():
    """
    Load camera transformation data from JSON file with caching.
    
    Returns:
        Dictionary with camera transformation data
    """
    global _CAMERA_TRANSFORM_CACHE, _CAMERA_TRANSFORM_LOCK
    
    with _CAMERA_TRANSFORM_LOCK:
        if _CAMERA_TRANSFORM_CACHE is None:
            try:
                with open('data/per_cam_transforms.json', 'r') as f:
                    _CAMERA_TRANSFORM_CACHE = json.load(f)
                logger.info(f"Loaded and cached camera transformation data with {len(_CAMERA_TRANSFORM_CACHE.get('frames', []))} total cameras")
            except FileNotFoundError:
                logger.error("Error: Could not find the per_cam_transforms.json file.")
                _CAMERA_TRANSFORM_CACHE = {'frames': []}
            except json.JSONDecodeError:
                logger.error("Error: Invalid JSON in per_cam_transforms.json file.")
                _CAMERA_TRANSFORM_CACHE = {'frames': []}
        
        return _CAMERA_TRANSFORM_CACHE


def get_frame_from_cache(camera_name, frame_number):
    """
    Get a frame from the cache if available
    
    Args:
        camera_name: Camera name/identifier
        frame_number: Frame number
        
    Returns:
        Cached frame or None if not found
    """
    global _FRAME_CACHE, _FRAME_CACHE_LOCK, _FRAME_CACHE_ACCESS_TIMES
    
    cache_key = f"{camera_name}_{frame_number}"
    
    with _FRAME_CACHE_LOCK:
        if cache_key in _FRAME_CACHE:
            # Update access time (LRU tracking)
            _FRAME_CACHE_ACCESS_TIMES[cache_key] = time.time()
            return _FRAME_CACHE[cache_key]
    
    return None


def add_frame_to_cache(camera_name, frame_number, frame):
    """
    Add a frame to the cache with LRU eviction if necessary
    
    Args:
        camera_name: Camera name/identifier
        frame_number: Frame number
        frame: Frame image data
    """
    global _FRAME_CACHE, _FRAME_CACHE_LOCK, _FRAME_CACHE_ACCESS_TIMES, _FRAME_CACHE_MAX_SIZE
    
    cache_key = f"{camera_name}_{frame_number}"
    
    with _FRAME_CACHE_LOCK:
        # Check if we need to evict frames
        if len(_FRAME_CACHE) >= _FRAME_CACHE_MAX_SIZE:
            # Find least recently used frames
            oldest_access = None
            oldest_key = None
            
            for key, access_time in _FRAME_CACHE_ACCESS_TIMES.items():
                if oldest_access is None or access_time < oldest_access:
                    oldest_access = access_time
                    oldest_key = key
            
            # Evict the oldest frame
            if oldest_key:
                del _FRAME_CACHE[oldest_key]
                del _FRAME_CACHE_ACCESS_TIMES[oldest_key]
        
        # Add new frame to cache
        _FRAME_CACHE[cache_key] = frame
        _FRAME_CACHE_ACCESS_TIMES[cache_key] = time.time()


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
    import time
    from concurrent.futures import as_completed
    
    logger.info(f"Preloading camera frames for frame {frame_number}")
    
    # Create a thread-safe dictionary
    preloaded_frames = {}
    
    # Get all frames from camera data
    all_frames = camera_data.get('frames', [])
    valid_camera_frames = []
    
    # First identify all valid frames to load
    for frame_info in all_frames:
        file_path = frame_info.get('file_path', '')
        
        # Skip if no file path or if it's a global camera (handled separately)
        if not file_path or 'global_' in file_path:
            continue
        
        # Check if frame is already in cache
        camera_name = os.path.basename(file_path)
        cached_frame = get_frame_from_cache(camera_name, frame_number)
        
        if cached_frame is not None:
            # Use cached frame
            preloaded_frames[file_path] = cached_frame
            continue
            
        # Try multiple possible locations for the image
        possible_paths = [
            os.path.join('data', file_path, f"frame_{frame_number}.jpg"),  # data/Camera_X/frame_N.jpg
            os.path.join(file_path, f"frame_{frame_number}.jpg"),          # Camera_X/frame_N.jpg
            os.path.join('output_frames', file_path, f"frame_{frame_number}.jpg")  # Legacy path
        ]
        
        # Check each possible location
        for image_path in possible_paths:
            if os.path.exists(image_path):
                valid_camera_frames.append((file_path, image_path, camera_name))
                break
    
    logger.info(f"Found {len(valid_camera_frames) + len(preloaded_frames)} valid camera frames ({len(preloaded_frames)} from cache)")
    
    # Define function to load a single image
    def load_single_image(args):
        file_path, image_path, camera_name = args
        
        try:
            frame_img = cv2.imread(image_path)
            if frame_img is not None and frame_img.size > 0:
                # Add to preloaded frames
                preloaded_frames[file_path] = frame_img
                
                # Add to cache for future use
                add_frame_to_cache(camera_name, frame_number, frame_img)
                
                return True, file_path
            else:
                logger.warning(f"Failed to read image for frame {frame_number} from camera {camera_name}")
                return False, file_path
        except Exception as e:
            logger.error(f"Error loading frame image from {camera_name}: {e}")
            return False, file_path
    
    # Load images in parallel
    start_time = time.time()
    load_success_count = 0
    
    # If we have no valid frames, return preloaded with cached frames
    if not valid_camera_frames:
        # If using only cached frames, still log that we found them
        if preloaded_frames:
            logger.info(f"Using {len(preloaded_frames)} cached frames for frame {frame_number}")
        else:
            logger.warning(f"No valid camera frames found for frame {frame_number}")
        return preloaded_frames
    
    # Use a reusable thread pool for image loading
    executor = get_thread_pool('image_loading', max_workers=8)
    
    future_to_path = {executor.submit(load_single_image, args): args[0]
                      for args in valid_camera_frames}
    
    for future in as_completed(future_to_path):
        try:
            success, file_path = future.result()
            if success:
                load_success_count += 1
        except Exception as e:
            logger.error(f"Unhandled exception in parallel image loading: {e}")
    
    load_time = time.time() - start_time
    logger.info(f"Preloaded {load_success_count}/{len(valid_camera_frames)} new camera frames in {load_time:.2f}s")
    
    return preloaded_frames


def preload_all_frames(start_frame: int, end_frame: int) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Preload all camera frames for a range of frame numbers.
    
    Args:
        start_frame: Starting frame number to preload
        end_frame: Ending frame number to preload
        
    Returns:
        Dictionary mapping frame numbers to dictionaries of camera frames
    """
    import os
    import cv2
    import time
    from concurrent.futures import as_completed
    
    logger.info(f"Preloading all camera frames for frames {start_frame} to {end_frame}")
    
    # Get camera transformation data
    camera_data = load_camera_transforms_cached()
    
    # Create frame dictionary
    all_frames_dict = {}
    
    # Track loading stats
    total_frames_to_load = (end_frame - start_frame + 1)
    total_cameras = len([f for f in camera_data.get('frames', []) if f.get('file_path') and 'global_' not in f.get('file_path', '')])
    total_images = total_frames_to_load * total_cameras
    loaded_images = 0
    
    start_time = time.time()
    
    # Define a function to load a single frame for all cameras
    def load_frame_for_all_cameras(frame_number):
        # Create a dictionary to store camera frames for this frame number
        frame_dict = {}
        
        # Get camera data for all cameras
        for frame_info in camera_data.get('frames', []):
            file_path = frame_info.get('file_path', '')
            
            # Skip if no file path or if it's a global camera
            if not file_path or 'global_' in file_path:
                continue
            
            # Check if frame is already in cache
            camera_name = os.path.basename(file_path)
            cached_frame = get_frame_from_cache(camera_name, frame_number)
            
            if cached_frame is not None:
                # Use cached frame
                frame_dict[file_path] = cached_frame
                continue
            
            # Try multiple possible locations for the image
            possible_paths = [
                os.path.join('data', file_path, f"frame_{frame_number}.jpg"),
                os.path.join(file_path, f"frame_{frame_number}.jpg"),
                os.path.join('output_frames', file_path, f"frame_{frame_number}.jpg")
            ]
            
            # Check each possible location
            for image_path in possible_paths:
                if os.path.exists(image_path):
                    try:
                        frame_img = cv2.imread(image_path)
                        if frame_img is not None and frame_img.size > 0:
                            # Add to frame dictionary
                            frame_dict[file_path] = frame_img
                            
                            # Add to cache for future use
                            add_frame_to_cache(camera_name, frame_number, frame_img)
                            break
                    except Exception as e:
                        logger.error(f"Error loading image {image_path}: {e}")
        
        return frame_number, frame_dict
    
    # Use a reusable thread pool for parallel frame loading
    executor = get_thread_pool('frame_loading', max_workers=8)
    
    # Create a list of frame numbers to process
    frame_range = range(start_frame, end_frame + 1)
    
    future_to_frame = {executor.submit(load_frame_for_all_cameras, frame_num): frame_num for frame_num in frame_range}
    
    for future in as_completed(future_to_frame):
        try:
            frame_num, frame_dict = future.result()
            all_frames_dict[frame_num] = frame_dict
            loaded_images += len(frame_dict)
            
            # Log progress periodically
            if len(all_frames_dict) % 10 == 0 or len(all_frames_dict) == total_frames_to_load:
                progress = (len(all_frames_dict) / total_frames_to_load) * 100
                logger.info(f"Preloaded {len(all_frames_dict)}/{total_frames_to_load} frames ({progress:.1f}%), {loaded_images} total images")
        except Exception as e:
            frame_num = future_to_frame[future]
            logger.error(f"Error preloading frame {frame_num}: {e}")
    
    load_time = time.time() - start_time
    logger.info(f"Preloaded {len(all_frames_dict)}/{total_frames_to_load} frames ({loaded_images} images) in {load_time:.2f}s")
    
    return all_frames_dict
