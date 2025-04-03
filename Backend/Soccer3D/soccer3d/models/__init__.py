"""
Models package for Soccer3D - handles detection and pose estimation.
"""

from soccer3d.models.detection import (
    initialize_triton_clients,
    check_models_ready,
    perform_yolo_inference_batched,
    warmup_triton_inference,
)

from soccer3d.models.pose import (
    initialize_mp_pose_pool,
    process_pose_from_detection,
    warmup_pytorch_cuda,
)

__all__ = [
    'initialize_triton_clients',
    'check_models_ready',
    'perform_yolo_inference_batched',
    'warmup_triton_inference',
    'initialize_mp_pose_pool',
    'process_pose_from_detection',
    'warmup_pytorch_cuda',
]
