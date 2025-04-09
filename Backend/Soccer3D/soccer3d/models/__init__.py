"""
Models package for Soccer3D - handles detection and pose estimation.
"""

from soccer3d.models.detection import (
    perform_yolo_inference_batched,
    warmup_pytorch_cuda,
    get_yolo_model,
    load_yolo_model,
    warmup_yolo_models,
)

from soccer3d.models.pose import (
    initialize_mp_pose_pool,
    process_pose_from_detection,
)

__all__ = [
    'perform_yolo_inference_batched',
    'warmup_pytorch_cuda',
    'get_yolo_model',
    'load_yolo_model',
    'warmup_yolo_models',
    'initialize_mp_pose_pool',
    'process_pose_from_detection',
]