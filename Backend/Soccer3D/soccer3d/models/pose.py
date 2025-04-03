"""
Pose estimation for Soccer3D.

This module handles pose estimation using MediaPipe.
"""
import os
import time
import warnings
import threading
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

# Import MediaPipe
try:
    import mediapipe as mp
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
    import mediapipe as mp

# Import PyTorch if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# MediaPipe pose keypoint names for more readable logging
POSE_KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

# Global variables
logger = logging.getLogger("Soccer3D")
MP_POSE_POOL = []
mp_pose_pool_lock = threading.RLock()
mp_pose_pool_index = 0
TIMING = {}

# Helper class to suppress output
class SuppressOutput:
    """Context manager to suppress output during MediaPipe initialization."""
    def __enter__(self):
        import sys
        import io
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
        
    def __exit__(self, *args):
        import sys
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def initialize_mp_pose_pool(config: Dict[str, Any]) -> bool:
    """
    Pre-load all MediaPipe pose models into a pool for parallel processing.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if successful, False otherwise
    """
    global MP_POSE_POOL, TIMING
    
    if 'pose_model_load' not in TIMING:
        TIMING['pose_model_load'] = 0.0
    if 'pool_initialization_time' not in TIMING:
        TIMING['pool_initialization_time'] = 0.0
    if 'pose_model_usages' not in TIMING:
        TIMING['pose_model_usages'] = [0] * config.get('mp_pose_pool_size', 20)
    
    start_time = time.time()
    pool_size = config.get('mp_pose_pool_size', 20)
    model_complexity = config.get('mp_pose_complexity', 0)
    
    logger.info(f"Pre-loading {pool_size} MediaPipe pose models into pool...")
    
    with mp_pose_pool_lock:
        # Check if pool is already initialized
        if MP_POSE_POOL:
            logger.info(f"MediaPipe pose model pool already initialized with {len(MP_POSE_POOL)} models")
            return True
            
        # Create the pool of pose models
        with warnings.catch_warnings(), SuppressOutput():
            warnings.simplefilter("ignore")
            for i in range(pool_size):
                model_start = time.time()
                try:
                    MP_POSE_POOL.append(mp.solutions.pose.Pose(
                        static_image_mode=True,
                        model_complexity=model_complexity,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    ))
                    model_time = time.time() - model_start
                    TIMING['pose_model_load'] += model_time
                    logger.info(f"Pose model {i+1}/{pool_size} loaded in {model_time:.2f}s")
                except Exception as e:
                    logger.error(f"Error loading MediaPipe pose model {i+1}: {e}")
    
    total_time = time.time() - start_time
    TIMING['pool_initialization_time'] = total_time
    
    if len(MP_POSE_POOL) > 0:
        logger.info(f"Successfully pre-loaded {len(MP_POSE_POOL)} MediaPipe pose models in {total_time:.2f}s")
        return True
    else:
        logger.error("Failed to initialize any MediaPipe pose models")
        return False


def get_mp_pose() -> Optional[mp.solutions.pose.Pose]:
    """
    Get MediaPipe pose model from the pre-loaded pool.
    
    Returns:
        MediaPipe pose instance from the model pool
    """
    global mp_pose_pool_index, MP_POSE_POOL, TIMING
    
    # Ensure the pool is initialized
    if not MP_POSE_POOL:
        with mp_pose_pool_lock:
            if not MP_POSE_POOL:
                logger.warning("MediaPipe pose model pool not initialized")
                return None
    
    # Get a model from the pool in a round-robin fashion
    with mp_pose_pool_lock:
        if not MP_POSE_POOL:
            # Handle error case if pool initialization failed
            logger.warning("MediaPipe pose model pool is empty")
            return None
        
        # Assign models round-robin
        model_index = mp_pose_pool_index
        model = MP_POSE_POOL[model_index]
        mp_pose_pool_index = (mp_pose_pool_index + 1) % len(MP_POSE_POOL)
        
        # Track usage count for this model
        if model_index < len(TIMING.get('pose_model_usages', [])):
            TIMING['pose_model_usages'][model_index] += 1
        
    return model


def process_pose_from_detection(
    frame_img: np.ndarray,
    detection: np.ndarray,
    intrinsics: Dict[str, float],
    transform_matrix: List[List[float]],
    player_id: int = 0,
    camera_name: str = "unknown",
    config: Dict[str, Any] = None
) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Process MediaPipe pose estimation on a cropped detection.
    
    Args:
        frame_img: Input image frame
        detection: Bounding box coordinates [x1, y1, x2, y2]
        intrinsics: Camera intrinsic parameters
        transform_matrix: Camera transformation matrix
        player_id: ID for the player being processed
        camera_name: Name of the camera being processed
        config: Configuration dictionary
        
    Returns:
        List of processed landmarks or None if detection failed
    """
    global TIMING
    
    # Set up timing if not already present
    if 'pose_estimation' not in TIMING:
        TIMING['pose_estimation'] = 0.0
    if 'pose_pure_inference' not in TIMING:
        TIMING['pose_pure_inference'] = 0.0
    if 'image_preprocessing' not in TIMING:
        TIMING['image_preprocessing'] = 0.0
    if 'result_postprocessing' not in TIMING:
        TIMING['result_postprocessing'] = 0.0
    
    padding = config.get('pose_from_detection_padding', 20) if config else 20
    
    try:
        # Get MediaPipe Pose instance
        mp_pose = get_mp_pose()
        
        if mp_pose is None:
            logger.error("Failed to get MediaPipe pose model from pool")
            return None
        
        # Get detection box coordinates
        x1, y1, x2, y2 = detection
        
        # Ensure box coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Log detection box for debugging
        logger.debug(f"Processing pose for player {player_id} in camera {camera_name} with box: [{x1}, {y1}, {x2}, {y2}]")
        
        # Start timing for preprocessing
        preprocess_start_time = time.time()
        
        # Add padding to the crop
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame_img.shape[1], x2 + padding), min(frame_img.shape[0], y2 + padding)
        
        # Skip if box is too small after padding adjustments
        if x2 - x1 < 10 or y2 - y1 < 10:
            logger.warning(f"Detection box for player {player_id} in camera {camera_name} too small after padding, skipping pose estimation")
            return None
        
        # Crop the image to the detection
        cropped_img = frame_img[int(y1):int(y2), int(x1):int(x2)]
        
        # Check for empty crop
        if cropped_img.size == 0:
            logger.warning(f"Empty crop for player {player_id} in camera {camera_name}")
            return None
        
        # Save crop for debugging if there's a problem with pose detection
        if player_id == 0 and camera_name not in ['global_view1', 'global_view2']:
            crop_dir = "debug_crops"
            os.makedirs(crop_dir, exist_ok=True)
            crop_file = os.path.join(crop_dir, f"player_{player_id}_cam_{camera_name}.jpg")
            cv2.imwrite(crop_file, cropped_img)
            logger.debug(f"Saved debug crop to {crop_file}")
        
        # Convert to RGB for MediaPipe (efficient conversion)
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        
        # End timing for preprocessing
        preprocess_time = time.time() - preprocess_start_time
        TIMING['image_preprocessing'] += preprocess_time
        
        # Process the cropped image
        # Start timing for pure inference
        pure_inference_start = time.time()
        results = mp_pose.process(cropped_rgb)
        pure_inference_time = time.time() - pure_inference_start
        TIMING['pose_pure_inference'] += pure_inference_time
        
        # Start timing for postprocessing
        postprocess_start_time = time.time()
        
        if results.pose_landmarks:
            # Adjust landmark coordinates back to original image space
            adjusted_landmarks = []
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # Convert relative coordinates to absolute
                x = landmark.x * (x2 - x1) + x1
                y = landmark.y * (y2 - y1) + y1
                z = landmark.z  # Z remains relative
                visibility = landmark.visibility
                
                # Log key landmarks
                if i in [11, 12, 23, 24]:  # shoulders and hips
                    keypoint_name = POSE_KEYPOINT_NAMES.get(i, f"unknown_{i}")
                    logger.debug(f"Keypoint {keypoint_name} (ID {i}) visibility: {visibility:.2f}, pos: [{x:.1f}, {y:.1f}]")
                
                adjusted_landmarks.append((x, y, z, visibility))
            
            # End timing for postprocessing
            postprocess_time = time.time() - postprocess_start_time
            TIMING['result_postprocessing'] += postprocess_time
            
            # Total pose processing time (includes preprocessing, inference, and postprocessing)
            total_pose_time = preprocess_time + pure_inference_time + postprocess_time
            TIMING['pose_estimation'] += total_pose_time
            
            logger.debug(f"Extracted {len(adjusted_landmarks)} pose landmarks for player {player_id} from camera {camera_name} in {total_pose_time:.3f}s")
            return adjusted_landmarks
        
        # End timing for postprocessing (even if no landmarks found)
        postprocess_time = time.time() - postprocess_start_time
        TIMING['result_postprocessing'] += postprocess_time
        
        # Total pose processing time (includes preprocessing, inference, and postprocessing)
        total_pose_time = preprocess_time + pure_inference_time + postprocess_time
        TIMING['pose_estimation'] += total_pose_time
        
        logger.debug(f"No pose landmarks detected for player {player_id} in camera {camera_name}")
        return None
    except Exception as e:
        logger.error(f"Error in pose detection for player {player_id} in camera {camera_name}: {e}")
        return None


def warmup_pytorch_cuda() -> bool:
    """
    Perform warm-up operations to initialize PyTorch CUDA runtime.
    This ensures that the first actual inference doesn't include CUDA initialization time.
    
    Returns:
        True if successful, False otherwise
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not available for CUDA warm-up")
        return False
    
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for warm-up")
            return False
            
        logger.info("Warming up PyTorch CUDA...")
        start_time = time.time()
        
        # Create dummy batch of appropriate size
        dummy_batch = torch.zeros((4, 3, 480, 640), device="cuda")
        
        # Run typical operations that will be used during preprocessing
        # to ensure CUDA kernels are compiled
        import torchvision.transforms.functional as F
        resized = F.resize(dummy_batch, [640, 640])
        normalized = resized / 255.0
        
        # Force synchronization to ensure all CUDA operations complete
        torch.cuda.synchronize()
        
        # Run dummy inference through network to initialize CUDA graphs
        dummy_output = normalized.cpu().numpy()
        
        warmup_time = time.time() - start_time
        logger.info(f"PyTorch CUDA warm-up completed in {warmup_time:.3f}s")
        return True
    except Exception as e:
        logger.warning(f"Error during PyTorch CUDA warm-up: {e}")
        return False
