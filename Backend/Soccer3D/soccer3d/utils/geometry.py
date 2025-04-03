"""
Geometry utilities for Soccer3D.

This module handles 3D geometry calculations, including ray intersection,
triangulation, and orientation computation.
"""
# Standard library imports
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np

# Local imports
from soccer3d.models.pose import POSE_KEYPOINT_NAMES

logger = logging.getLogger("Soccer3D")

# Global timing stats
TIMING = {
    'triangulation': 0.0,
    'orientation_calc': 0.0,
    'parallel_ray_processing': 0.0,
}


def find_ray_intersection(rays: List[Tuple[np.ndarray, np.ndarray]], entity_name: str = "unknown", config: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """
    Find the intersection point of multiple rays using least squares with efficient parallel ray processing.
    
    Args:
        rays: List of (origin, direction) tuples for each ray
        entity_name: Name of the entity being triangulated for logging
        config: Configuration dictionary
        
    Returns:
        Intersection point as numpy array, or None if not found
    """
    start_time = time.time()  # Add timing measurement
    
    if len(rays) < 2:
        logger.warning(f"Not enough rays to triangulate {entity_name} position (got {len(rays)}, need at least 2)")
        return None
    
    logger.debug(f"Triangulating {entity_name} position using {len(rays)} rays")
    
    try:
        # Process rays in parallel for better performance with large numbers of rays
        def process_ray(ray_with_index):
            i, (origin, direction) = ray_with_index
            
            # Verify data types and shapes
            if not isinstance(origin, np.ndarray) or not isinstance(direction, np.ndarray):
                origin = np.array(origin)
                direction = np.array(direction)
            
            # Ensure direction is normalized
            direction = direction / np.linalg.norm(direction)
            
            # Create skew-symmetric matrix for cross product
            skew = np.array([
                [0, -direction[2], direction[1]],
                [direction[2], 0, -direction[0]],
                [-direction[1], direction[0], 0]
            ])
            
            # Calculate the corresponding right-hand side
            b_vector = np.cross(direction, origin)
            
            return (i, skew, b_vector)
        
        # Determine if we should use parallel processing
        # Only use parallel processing if we have enough rays to make it worthwhile
        max_workers = config.get('max_workers', 8) if config else 8
        use_parallel = len(rays) >= 10 and max_workers > 1
        
        # Allocate arrays outside the parallel section
        A = np.zeros((len(rays) * 3, 3))
        b = np.zeros(len(rays) * 3)
        
        if use_parallel:
            # Prepare data for parallel processing
            ray_items = [(i, ray) for i, ray in enumerate(rays)]
            
            # Process rays in parallel
            parallel_start_time = time.time()
            with ThreadPoolExecutor(max_workers=min(max_workers, len(rays))) as executor:
                # Submit all ray processing tasks and collect futures
                futures = [executor.submit(process_ray, item) for item in ray_items]
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        i, skew, b_vector = future.result()
                        # Add to system matrix and right-hand side
                        A[i*3:(i+1)*3] = skew
                        b[i*3:(i+1)*3] = b_vector
                    except Exception as e:
                        logger.error(f"Error in parallel ray processing: {e}")
            
            parallel_time = time.time() - parallel_start_time
            if 'parallel_ray_processing' in TIMING:
                TIMING['parallel_ray_processing'] += parallel_time
            else:
                TIMING['parallel_ray_processing'] = parallel_time
                
            logger.debug(f"Parallel ray processing for {entity_name} with {len(rays)} rays took {parallel_time:.3f}s")
        else:
            # Use the sequential approach for small numbers of rays
            for i, (origin, direction) in enumerate(rays):
                # Verify data types and shapes
                if not isinstance(origin, np.ndarray) or not isinstance(direction, np.ndarray):
                    origin = np.array(origin)
                    direction = np.array(direction)
                
                # Ensure direction is normalized
                direction = direction / np.linalg.norm(direction)
                
                # Create skew-symmetric matrix for cross product
                skew = np.array([
                    [0, -direction[2], direction[1]],
                    [direction[2], 0, -direction[0]],
                    [-direction[1], direction[0], 0]
                ])
                
                # Add to system matrix
                A[i*3:(i+1)*3] = skew
                b[i*3:(i+1)*3] = np.cross(direction, origin)
        
        # Solve using least squares (faster with SVD for better numerical stability)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        logger.debug(f"Triangulated {entity_name} position: [{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}]")
        
        # Track timing
        triangulation_time = time.time() - start_time
        TIMING['triangulation'] = TIMING.get('triangulation', 0.0) + triangulation_time
        
        return x
    except Exception as e:
        logger.error(f"Failed to triangulate {entity_name} position: {e}")
        # Track timing even for errors
        triangulation_time = time.time() - start_time
        TIMING['triangulation'] = TIMING.get('triangulation', 0.0) + triangulation_time
        return None


def triangulate_pose(pose_rays: Dict[int, List[Tuple[np.ndarray, np.ndarray]]], config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Triangulate 3D pose from rays for multiple keypoints and calculate player orientation.
    
    Args:
        pose_rays: Dictionary of keypoint index to list of rays
        config: Configuration dictionary
        
    Returns:
        Tuple of (pose_3d keypoints, orientation_vector, cardinal_direction)
    """
    pose_3d = {}
    
    logger.debug(f"Triangulating pose from {len(pose_rays)} keypoint types")
    
    # Log how many rays we have for key points
    key_points = [11, 12, 23, 24]  # shoulders and hips
    for kp in key_points:
        if kp in pose_rays:
            logger.debug(f"Keypoint {POSE_KEYPOINT_NAMES.get(kp, kp)} has {len(pose_rays[kp])} rays")
    
    triangulation_start_time = time.time()  # Track triangulation time
    
    try:
        # Process each keypoint in parallel but with limited concurrency
        max_workers = config.get('max_workers', 8) if config else 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_keypoint = {
                executor.submit(
                    find_ray_intersection, 
                    rays, 
                    f"keypoint_{POSE_KEYPOINT_NAMES.get(keypoint_idx, keypoint_idx)}",
                    config
                ): keypoint_idx
                for keypoint_idx, rays in pose_rays.items()
                if len(rays) >= 2  # Need at least 2 rays
            }
            
            for future in as_completed(future_to_keypoint):
                keypoint_idx = future_to_keypoint[future]
                try:
                    intersection_point = future.result()
                    
                    if intersection_point is not None:
                        pose_3d[keypoint_idx] = intersection_point
                except Exception as e:
                    logger.error(f"Error triangulating keypoint {keypoint_idx}: {e}")
        
        # Capture the full triangulation time
        triangulation_time = time.time() - triangulation_start_time
        TIMING['triangulation'] = TIMING.get('triangulation', 0.0) + triangulation_time
        
        logger.info(f"Successfully triangulated {len(pose_3d)}/{len(pose_rays)} pose keypoints")
        
        # Log each keypoint position with its name
        for keypoint_idx, position in pose_3d.items():
            keypoint_name = POSE_KEYPOINT_NAMES.get(keypoint_idx, f"unknown_{keypoint_idx}")
            logger.info(f"Keypoint {keypoint_name}: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        
        # Calculate player orientation if we have enough keypoints
        orientation_vector, cardinal_direction = calculate_player_orientation(pose_3d)
        if orientation_vector is not None:
            logger.info(f"Player orientation vector: [{orientation_vector[0]:.2f}, {orientation_vector[1]:.2f}, {orientation_vector[2]:.2f}]")
            logger.info(f"Player is facing: {cardinal_direction}")
        else:
            logger.warning("Could not determine player orientation")
    
    except Exception as e:
        logger.error(f"Error in pose triangulation: {e}")
        
        # Capture the full triangulation time even for errors
        triangulation_time = time.time() - triangulation_start_time
        TIMING['triangulation'] = TIMING.get('triangulation', 0.0) + triangulation_time
        
        orientation_vector, cardinal_direction = None, None
    
    return pose_3d, orientation_vector, cardinal_direction


def calculate_player_orientation(pose_3d: Dict[int, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Calculate the direction the player is facing based on pose keypoints.
    
    Args:
        pose_3d: Dictionary of keypoint index to 3D position
        
    Returns:
        Tuple of (unit vector indicating player facing direction, cardinal direction description)
    """
    start_time = time.time()  # Add timing measurement
    
    # Check if we have necessary keypoints for orientation calculation
    if 11 in pose_3d and 12 in pose_3d:  # Left and right shoulders
        # Use shoulder line to determine orientation
        left_shoulder = pose_3d[11]
        right_shoulder = pose_3d[12]
        line_vector = right_shoulder - left_shoulder
        
        # Create a normalized direction vector perpendicular to the shoulder line
        # in the XY plane (assuming Z is up per user's coordinate system)
        # The player faces perpendicular to the shoulder line
        # NOTE: Using positive sign for Y component to match field coordinate system
        facing = np.array([line_vector[1], -line_vector[0], 0])
        
        logger.debug(f"Using shoulders for orientation calculation: l={left_shoulder}, r={right_shoulder}")
        logger.debug(f"Shoulder line vector: {line_vector}")
        
    elif 23 in pose_3d and 24 in pose_3d:  # Left and right hips as fallback
        # Use hip line to determine orientation if shoulders aren't available
        left_hip = pose_3d[23]
        right_hip = pose_3d[24]
        line_vector = right_hip - left_hip
        
        # Create a normalized direction vector perpendicular to the hip line
        # in the XY plane (assuming Z is up per user's coordinate system)
        # NOTE: Using positive sign for Y component to match field coordinate system
        facing = np.array([line_vector[1], -line_vector[0], 0])
        
        logger.debug(f"Using hips for orientation calculation: l={left_hip}, r={right_hip}")
        logger.debug(f"Hip line vector: {line_vector}")
        
    # Fallback to a default orientation (facing up the field) with low confidence
    elif len(pose_3d) > 0:
        # If we have any keypoints but not shoulders or hips, use a default orientation
        logger.warning("Using fallback orientation (facing up field) based on limited keypoints")
        facing = np.array([0.0, 1.0, 0.0])  # Default to facing up the field
    else:
        logger.warning("Insufficient keypoints to determine player orientation")
        
        # Track timing even for errors
        orientation_time = time.time() - start_time
        TIMING['orientation_calc'] = TIMING.get('orientation_calc', 0.0) + orientation_time
        
        return None, None
    
    # Normalize to unit vector
    facing_norm = np.linalg.norm(facing)
    if facing_norm < 1e-6:
        logger.warning("Could not determine player orientation: zero-length vector")
        
        # Track timing even for errors
        orientation_time = time.time() - start_time
        TIMING['orientation_calc'] = TIMING.get('orientation_calc', 0.0) + orientation_time
        
        return None, None
    
    facing_unit = facing / facing_norm
    
    # Calculate the angle in the XY plane (assuming Z is up per user's coordinate system)
    # atan2(y,x) gives angle from positive x-axis in counterclockwise direction
    angle_rad = math.atan2(facing_unit[1], facing_unit[0])
    angle_deg = math.degrees(angle_rad)
    
    # Convert angle to cardinal direction with respect to field
    cardinal_direction = get_cardinal_direction(angle_deg)
    field_direction = get_field_direction(facing_unit)
    
    # Track timing
    orientation_time = time.time() - start_time
    TIMING['orientation_calc'] = TIMING.get('orientation_calc', 0.0) + orientation_time
    
    return facing_unit, f"{cardinal_direction} - {field_direction}"


def get_cardinal_direction(angle_deg: float) -> str:
    """
    Convert an angle in degrees to a cardinal direction description.
    
    Args:
        angle_deg: Angle in degrees (0 = east, 90 = north, etc.)
        
    Returns:
        String description of the cardinal direction
    """
    # Map the angle to 8 cardinal directions
    adjusted_angle = angle_deg % 360
    
    if 22.5 <= adjusted_angle < 67.5:
        return "northeast (NE)"
    elif 67.5 <= adjusted_angle < 112.5:
        return "north (N)"
    elif 112.5 <= adjusted_angle < 157.5:
        return "northwest (NW)"
    elif 157.5 <= adjusted_angle < 202.5:
        return "west (W)"
    elif 202.5 <= adjusted_angle < 247.5:
        return "southwest (SW)"
    elif 247.5 <= adjusted_angle < 292.5:
        return "south (S)"
    elif 292.5 <= adjusted_angle < 337.5:
        return "southeast (SE)"
    else:  # 337.5 <= adjusted_angle < 22.5
        return "east (E)"


def get_field_direction(facing_vector: np.ndarray) -> str:
    """
    Convert a facing vector to soccer field-specific direction.
    
    Args:
        facing_vector: Unit vector indicating player orientation
        
    Returns:
        String description of direction on soccer field
    """
    # Extract x and y components (side-to-side and goal-to-goal)
    x_component = facing_vector[0]  # side-to-side component
    y_component = facing_vector[1]  # goal-to-goal component
    
    # Determine the primary direction based on the strongest component
    if abs(x_component) > abs(y_component):
        # Primarily moving side to side
        if x_component > 0:
            return "facing right sideline"
        else:
            return "facing left sideline"
    else:
        # Primarily moving goal to goal
        if y_component > 0:
            return "facing far goal"
        else:
            return "facing near goal"


def compute_distance_between_points(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two 3D points.
    
    Args:
        point1: First 3D point
        point2: Second 3D point
        
    Returns:
        Distance between the points
    """
    return np.linalg.norm(point1 - point2)
