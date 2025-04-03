"""
Utility functions for Soccer3D.

This package contains various utility modules for camera transformations,
ray calculations, and 3D geometry.
"""

from soccer3d.utils.camera import (
    extract_camera_position,
    extract_camera_direction,
    validate_intrinsics,
)

from soccer3d.utils.ray import (
    get_ray_from_camera,
)

from soccer3d.utils.geometry import (
    find_ray_intersection,
    triangulate_pose,
    calculate_player_orientation,
    get_cardinal_direction,
    get_field_direction,
)

__all__ = [
    'extract_camera_position',
    'extract_camera_direction',
    'validate_intrinsics',
    'get_ray_from_camera',
    'find_ray_intersection',
    'triangulate_pose',
    'calculate_player_orientation',
    'get_cardinal_direction',
    'get_field_direction',
]
