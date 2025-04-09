"""
Detection models for Soccer3D.

This module implements player and ball detection using YOLO models directly.
"""
# Standard library imports
import time
import threading
import logging
import sys
import subprocess
import os
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import cv2

# Import supervision for detection processing
try:
    import supervision as sv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supervision"])
    import supervision as sv

# Import torch and YOLO
try:
    import torch
    import torchvision.transforms.functional as F
    from ultralytics import YOLO
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
# Global variables
logger = logging.getLogger("Soccer3D")
TIMING = {}

# Thread-safe locks
model_cache_lock = threading.RLock()

# Global model cache
YOLO_MODELS = {}


def get_yolo_model(model_path: str, config: Dict[str, Any]) -> Optional[YOLO]:
    """
    Get or initialize a YOLO model.
    
    Args:
        model_path: Path to YOLO model file
        config: Configuration dictionary
        
    Returns:
        YOLO model instance or None if failed
    """
    global YOLO_MODELS, model_cache_lock, TIMING
    
    # Set up timing if not already present
    if 'player_model_load' not in TIMING:
        TIMING['player_model_load'] = 0.0
    if 'ball_model_load' not in TIMING:
        TIMING['ball_model_load'] = 0.0
    
    model_key = model_path
    
    # Check if model is already loaded
    with model_cache_lock:
        if model_key in YOLO_MODELS:
            return YOLO_MODELS[model_key]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        start_time = time.time()
        
        # Load YOLO model
        model = YOLO(model_path, task='detect')
        
        # Apply FP16 precision if configured
        if 'model_precision' in config and config['model_precision'] == 'fp16' and torch.cuda.is_available():
            logger.info(f"Loading model {model_path} with FP16 precision")
            model.to('cuda').half()
        elif torch.cuda.is_available():
            logger.info(f"Loading model {model_path} with FP32 precision")
            model.to('cuda')
        
        load_time = time.time() - start_time
        
        # Cache the model
        with model_cache_lock:
            YOLO_MODELS[model_key] = model
        
        # Update timing based on model type
        if "player" in model_path:
            TIMING['player_model_load'] = load_time
            logger.info(f"Player model loaded in {load_time:.2f}s")
        else:
            TIMING['ball_model_load'] = load_time
            logger.info(f"Ball model loaded in {load_time:.2f}s")
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to load YOLO model {model_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def perform_yolo_inference_batched(
    model_path: str, 
    input_images: List[np.ndarray],
    conf_threshold: float,
    config: Dict[str, Any],
    preprocessed_data: Optional[Tuple[np.ndarray, List[Tuple[int, int]]]] = None
) -> List[Optional[sv.Detections]]:
    """
    Perform batched inference using YOLO model directly.
    
    Args:
        model_path: Path to YOLO model file
        input_images: List of input images as numpy arrays
        conf_threshold: Confidence threshold for detections
        config: Configuration dictionary
        preprocessed_data: Optional preprocessed data (not used with direct YOLO implementation)
        
    Returns:
        List of Supervision Detections objects or None for each input
    """
    global TIMING
    
    # Set up timing if not already present
    if 'player_detection' not in TIMING:
        TIMING['player_detection'] = 0.0
    if 'ball_detection' not in TIMING:
        TIMING['ball_detection'] = 0.0
    if 'player_pure_inference' not in TIMING:
        TIMING['player_pure_inference'] = 0.0
    if 'ball_pure_inference' not in TIMING:
        TIMING['ball_pure_inference'] = 0.0
    
    try:
        # Skip empty batches
        if not input_images:
            return []
        
        # Get YOLO model
        model = get_yolo_model(model_path, config)
        
        if model is None:
            logger.error(f"YOLO model not available for {model_path}")
            return [None] * len(input_images)
        
        batch_size = len(input_images)
        logger.debug(f"Processing batch of size {batch_size} with model {model_path}")
        
        # Start timing for pure inference
        pure_inference_start = time.time()
        
        # Perform inference with the YOLO model
        # Note: YOLO already handles preprocessing internally, so we pass the original images
        results = model(input_images, conf=conf_threshold, verbose=False)
        
        # Get pure inference time
        pure_inference_time = time.time() - pure_inference_start
        
        # Update pure inference timing metrics based on model type
        if "player" in model_path:
            TIMING['player_pure_inference'] += pure_inference_time
            logger.debug(f"Pure player inference time for batch of {batch_size} images: {pure_inference_time:.3f}s")
        elif "ball" in model_path:
            TIMING['ball_pure_inference'] += pure_inference_time
            logger.debug(f"Pure ball inference time for batch of {batch_size} images: {pure_inference_time:.3f}s")
        
        # Convert YOLO results to Supervision Detections format
        all_detections = []
        
        for i, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                # Extract boxes
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Skip if no valid detections
                if len(boxes) == 0:
                    all_detections.append(sv.Detections(
                        xyxy=np.zeros((0, 4)),
                        confidence=np.array([]),
                        class_id=np.array([]),
                    ))
                    continue
                
                # Create Supervision Detection object
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=confidences,
                    class_id=class_ids,
                )
                
                all_detections.append(detections)
                
                logger.debug(f"Batch item {i} detected {len(boxes)} objects with confidence >= {conf_threshold}")
            else:
                # No detections for this image
                all_detections.append(sv.Detections(
                    xyxy=np.zeros((0, 4)),
                    confidence=np.array([]),
                    class_id=np.array([]),
                ))
                logger.debug(f"Batch item {i} detected no objects")
        
        # Update total timing metrics based on model type
        total_inference_time = pure_inference_time
        if "player" in model_path:
            TIMING['player_detection'] += total_inference_time
            logger.debug(f"Batched player detection ({batch_size} images) completed in {total_inference_time:.3f}s using {model_path}")
        elif "ball" in model_path:
            TIMING['ball_detection'] += total_inference_time
            logger.debug(f"Batched ball detection ({batch_size} images) completed in {total_inference_time:.3f}s using {model_path}")
        
        return all_detections
        
    except Exception as e:
        logger.error(f"Error performing batched YOLO inference with model at {model_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [None] * len(input_images)


def warmup_pytorch_cuda():
    """
    Warm up PyTorch/CUDA to ensure that the first inference is not slow.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        logger.warning("CUDA not available for warmup")
        return
    
    try:
        logger.info("Warming up PyTorch CUDA...")
        start_time = time.time()
        
        # Create a small dummy batch for warm-up
        dummy_tensor = torch.zeros((1, 3, 640, 640), device='cuda')
        dummy_tensor = dummy_tensor * 2.0  # Simple operation to force CUDA initialization
        
        # Ensure synchronization
        torch.cuda.synchronize()
        
        warm_up_time = time.time() - start_time
        logger.info(f"PyTorch CUDA warm-up completed in {warm_up_time:.2f}s")
    except Exception as e:
        logger.warning(f"Error during PyTorch CUDA warm-up: {e}")


def load_yolo_model(model_path: str) -> Optional[YOLO]:
    """
    Load a YOLO model with error handling.
    
    Args:
        model_path: Path to YOLO model file
        
    Returns:
        YOLO model or None if failed
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        # Load YOLO model
        model = YOLO(model_path)
        logger.info(f"Loaded YOLO model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None


def warmup_yolo_models(config: Dict[str, Any]) -> bool:
    """
    Preload and warm up YOLO models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure PyTorch/CUDA is warmed up
        warmup_pytorch_cuda()
        
        # Preload player model
        if 'player_model_path' in config:
            player_model = get_yolo_model(config['player_model_path'], config)
            if player_model is None:
                logger.warning("Failed to warm up player model")
        
        # Preload ball model
        if 'ball_model_path' in config:
            ball_model = get_yolo_model(config['ball_model_path'], config)
            if ball_model is None:
                logger.warning("Failed to warm up ball model")
        
        return True
    except Exception as e:
        logger.error(f"Error warming up YOLO models: {e}")
        return False