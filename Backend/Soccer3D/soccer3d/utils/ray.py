"""
Ray calculation utilities for Soccer3D.

This module handles 3D ray calculations and caching.
"""
import time
import threading
import numpy as np
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger("Soccer3D")

# Global ray cache with lock
RAY_CACHE = {}
ray_cache_lock = threading.RLock()
MAX_RAY_CACHE_SIZE = 10000  # Default max cache size

# Global timing stats
TIMING = {
    'ray_calculation': 0.0,
    'ray_calculation_counts': 0,
    'ray_cache_hits': 0,
    'ray_cache_misses': 0,
}


def initialize_ray_cache(config: Dict[str, Any]) -> None:
    """
    Initialize the ray cache with configuration parameters.
    
    Args:
        config: Configuration dictionary
    """
    global MAX_RAY_CACHE_SIZE, RAY_CACHE
    
    # Reset the cache
    with ray_cache_lock:
        RAY_CACHE.clear()
    
    # Set max cache size from config
    MAX_RAY_CACHE_SIZE = config.get('max_ray_cache_size', 10000)
    
    logger.info(f"Ray cache initialized with max size: {MAX_RAY_CACHE_SIZE}")


def get_ray_from_camera(
    camera_pos_tuple: Tuple[float, float, float],
    camera_dir_tuple: Tuple[float, float, float],
    point_2d_tuple: Tuple[float, float],
    intrinsics_tuple: Tuple[float, float, float, float, float, float],
    transform_matrix_tuple: Tuple[Tuple[float, float, float, float], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 2D image point to 3D ray from camera using camera parameters with caching.
    
    Args:
        camera_pos_tuple: Camera position as a tuple for caching
        camera_dir_tuple: Camera direction as a tuple for caching
        point_2d_tuple: 2D point coordinates as a tuple for caching
        intrinsics_tuple: Camera intrinsics as a tuple for caching
        transform_matrix_tuple: Camera transform matrix as a tuple of tuples for caching
        
    Returns:
        Tuple of (camera position, ray direction)
    """
    global RAY_CACHE, MAX_RAY_CACHE_SIZE, TIMING
    
    # Round floating point values to improve cache hit rate (fix precision issues)
    # Only round for the cache key, not for actual calculation
    precision = 3  # Round to 3 decimal places - adjust as needed for precision/hit rate balance
    
    # Create a rounded version of position and point for cache key
    pos_rounded = tuple(round(v, precision) for v in camera_pos_tuple)
    point_2d_rounded = tuple(round(v, precision) for v in point_2d_tuple)
    
    # Only use essential parts of intrinsics, rounded for better matching
    intrinsics_essential = tuple(round(v, precision) for v in intrinsics_tuple[:4])
    
    # Create a cache key from the rounded input parameters
    cache_key = (pos_rounded, point_2d_rounded, intrinsics_essential)
    
    # Check if we have a cached result
    with ray_cache_lock:
        if cache_key in RAY_CACHE:
            # Move this entry to the end of the cache (most recently used)
            result = RAY_CACHE.pop(cache_key)
            RAY_CACHE[cache_key] = result
            TIMING['ray_cache_hits'] += 1
            return result
        TIMING['ray_cache_misses'] += 1
    
    start_time = time.time()  # Add timing measurement
    try:
        # Convert tuples back to numpy arrays
        camera_pos = np.array(camera_pos_tuple)
        transform_matrix = np.array(transform_matrix_tuple)
        point_2d = np.array(point_2d_tuple)
        
        # Recreate intrinsics dictionary
        w, h, fl_x, fl_y, cx, cy = intrinsics_tuple
        intrinsics = {'w': w, 'h': h, 'fl_x': fl_x, 'fl_y': fl_y, 'cx': cx, 'cy': cy}
        
        # Get camera rotation from transform matrix
        camera_rotation = transform_matrix[:3, :3]
        
        # Normalize the 2D point to camera coordinates
        x = (point_2d[0] - intrinsics['cx']) / intrinsics['fl_x']
        y = (point_2d[1] - intrinsics['cy']) / intrinsics['fl_y']
        
        # Create ray direction in camera space
        ray_dir_cam = np.array([x, y, -1.0])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
        
        # Convert to world space using camera rotation
        ray_dir_world = camera_rotation @ ray_dir_cam
        
        # Create result tuple
        result = (camera_pos, ray_dir_world)
        
        # Cache the result with LRU eviction strategy
        with ray_cache_lock:
            # Limit cache size
            if len(RAY_CACHE) >= MAX_RAY_CACHE_SIZE:
                # Remove the oldest key (first item in the dict) - Python 3.7+ preserves insertion order
                oldest_key = next(iter(RAY_CACHE))
                RAY_CACHE.pop(oldest_key)
                
            # Add new entry at the end (most recently used)
            RAY_CACHE[cache_key] = result
            
            # Log cache performance periodically
            if (TIMING['ray_cache_hits'] + TIMING['ray_cache_misses']) % 100 == 0:
                total_requests = TIMING['ray_cache_hits'] + TIMING['ray_cache_misses']
                hit_rate = TIMING['ray_cache_hits'] / total_requests * 100 if total_requests > 0 else 0
                logger.debug(f"Ray cache performance: {hit_rate:.1f}% hit rate ({TIMING['ray_cache_hits']}/{total_requests}), size: {len(RAY_CACHE)}/{MAX_RAY_CACHE_SIZE}")
        
        ray_time = time.time() - start_time  # Calculate elapsed time
        TIMING['ray_calculation'] += ray_time  # Add to total ray calculation time
        TIMING['ray_calculation_counts'] += 1  # Increment counter
        
        return result
    except Exception as e:
        logger.error(f"Error in ray calculation: {e}")
        ray_time = time.time() - start_time  # Still track time even for errors
        TIMING['ray_calculation'] += ray_time  # Add to total ray calculation time
        TIMING['ray_calculation_counts'] += 1  # Increment counter
        # Return safe default values
        return np.array(camera_pos_tuple), np.array([0.0, 0.0, -1.0])


def convert_2d_to_3d_ray(
    point_2d: np.ndarray,
    intrinsics: Dict[str, float],
    transform_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D point to a 3D ray without caching.
    
    Args:
        point_2d: 2D point coordinates as numpy array
        intrinsics: Camera intrinsic parameters
        transform_matrix: Camera transformation matrix
        
    Returns:
        Tuple of (camera position, ray direction)
    """
    # Extract camera position from transform matrix
    camera_pos = np.array([transform_matrix[0, 3], transform_matrix[1, 3], transform_matrix[2, 3]])
    
    # Get camera rotation from transform matrix
    camera_rotation = transform_matrix[:3, :3]
    
    # Normalize the 2D point to camera coordinates
    x = (point_2d[0] - intrinsics['cx']) / intrinsics['fl_x']
    y = (point_2d[1] - intrinsics['cy']) / intrinsics['fl_y']
    
    # Create ray direction in camera space
    ray_dir_cam = np.array([x, y, -1.0])
    ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
    
    # Convert to world space using camera rotation
    ray_dir_world = camera_rotation @ ray_dir_cam
    
    return camera_pos, ray_dir_world


def get_ray_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the ray cache.
    
    Returns:
        Dictionary with cache statistics
    """
    global RAY_CACHE, TIMING
    
    with ray_cache_lock:
        cache_size = len(RAY_CACHE)
        hits = TIMING['ray_cache_hits']
        misses = TIMING['ray_cache_misses']
        total = hits + misses
        hit_rate = hits / total * 100 if total > 0 else 0
        
        return {
            'cache_size': cache_size,
            'max_cache_size': MAX_RAY_CACHE_SIZE,
            'cache_usage': cache_size / MAX_RAY_CACHE_SIZE * 100,
            'hits': hits,
            'misses': misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'ray_calculation_time': TIMING['ray_calculation'],
            'ray_calculation_count': TIMING['ray_calculation_counts'],
            'avg_ray_calculation_time': TIMING['ray_calculation'] / TIMING['ray_calculation_counts'] if TIMING['ray_calculation_counts'] > 0 else 0,
        }
