"""
Detection models for Soccer3D.

This module handles player and ball detection using Triton Inference Server.
"""
# Standard library imports
import time
import threading
import logging
import sys
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import cv2

# Try to import tritonclient for direct inference
try:
    import tritonclient.grpc as httpclient
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tritonclient[all]"])
    import tritonclient.grpc as httpclient

# Import supervision for detection processing if available
try:
    import supervision as sv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supervision"])
    import supervision as sv

# Import torch for GPU acceleration if available
try:
    import torch
    import torchvision.transforms.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Global variables
logger = logging.getLogger("Soccer3D")
TIMING = {}
MODEL_LOADED_CACHE = {'player_detector': False, 'ball_detector': False}
TRITON_CLIENTS = {'player_detector': None, 'ball_detector': None}

# Thread-safe locks
triton_client_lock = threading.RLock()
model_cache_lock = threading.RLock()


def initialize_triton_clients(config: Dict[str, Any]) -> bool:
    """
    Pre-initialize connections to Triton inference servers.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if successful, False otherwise
    """
    global TRITON_CLIENTS, MODEL_LOADED_CACHE, TIMING
    
    if 'triton_client_initialization' not in TIMING:
        TIMING['triton_client_initialization'] = 0.0
    
    start_time = time.time()
    logger.info("Initializing connections to Triton inference servers...")
    
    with triton_client_lock:
        # Initialize player detector client
        player_server_url = config['player_model_url'].split("/")[2]
        try:
            logger.info(f"Connecting to player detector server at {player_server_url}")
            player_client = httpclient.InferenceServerClient(
                url=player_server_url,
                verbose=False,
                ssl=False,
            )
            
            # Verify connection
            if not player_client.is_server_live():
                logger.error(f"Triton server at {player_server_url} is not live")
                TRITON_CLIENTS['player_detector'] = None
            elif not player_client.is_server_ready():
                logger.error(f"Triton server at {player_server_url} is not ready")
                TRITON_CLIENTS['player_detector'] = None
            else:
                # Extract player model name from URL
                player_model_url_parts = config['player_model_url'].split("/")
                player_model_name = player_model_url_parts[-1]
                
                if not player_client.is_model_ready(player_model_name):
                    logger.error(f"Player model {player_model_name} is not ready")
                    TRITON_CLIENTS['player_detector'] = None
                else:
                    logger.info(f"Successfully connected to player detector server")
                    TRITON_CLIENTS['player_detector'] = player_client
                    with model_cache_lock:
                        MODEL_LOADED_CACHE['player_detector'] = True
        except Exception as e:
            logger.error(f"Failed to connect to player detector server: {e}")
            TRITON_CLIENTS['player_detector'] = None
        
        # Initialize ball detector client
        ball_server_url = config['ball_model_url'].split("/")[2]
        try:
            logger.info(f"Connecting to ball detector server at {ball_server_url}")
            ball_client = httpclient.InferenceServerClient(
                url=ball_server_url,
                verbose=False,
                ssl=False,
            )
            
            # Verify connection
            if not ball_client.is_server_live():
                logger.error(f"Triton server at {ball_server_url} is not live")
                TRITON_CLIENTS['ball_detector'] = None
            elif not ball_client.is_server_ready():
                logger.error(f"Triton server at {ball_server_url} is not ready")
                TRITON_CLIENTS['ball_detector'] = None
            else:
                # Extract ball model name from URL
                ball_model_url_parts = config['ball_model_url'].split("/")
                ball_model_name = ball_model_url_parts[-1]
                
                if not ball_client.is_model_ready(ball_model_name):
                    logger.error(f"Ball model {ball_model_name} is not ready")
                    TRITON_CLIENTS['ball_detector'] = None
                else:
                    logger.info(f"Successfully connected to ball detector server")
                    TRITON_CLIENTS['ball_detector'] = ball_client
                    with model_cache_lock:
                        MODEL_LOADED_CACHE['ball_detector'] = True
        except Exception as e:
            logger.error(f"Failed to connect to ball detector server: {e}")
            TRITON_CLIENTS['ball_detector'] = None
    
    total_time = time.time() - start_time
    TIMING['triton_client_initialization'] = total_time
    logger.info(f"Triton client initialization completed in {total_time:.2f}s")
    
    # Log connection status
    connection_status = {
        'player_detector': TRITON_CLIENTS['player_detector'] is not None,
        'ball_detector': TRITON_CLIENTS['ball_detector'] is not None
    }
    logger.info(f"Triton connection status: {connection_status}")
    
    return all(connection_status.values())


def get_triton_client(model_url: str, config: Dict[str, Any]) -> Optional[httpclient.InferenceServerClient]:
    """
    Get a pre-initialized Triton Client connected to Triton Server.
    
    Args:
        model_url: URL to the Triton server model endpoint
        config: Configuration dictionary
        
    Returns:
        A pre-initialized instance of InferenceServerClient or None if failed
    """
    global TRITON_CLIENTS, MODEL_LOADED_CACHE, TIMING
    
    # Set up timing if not already present
    if 'player_model_load' not in TIMING:
        TIMING['player_model_load'] = 0.0
    if 'ball_model_load' not in TIMING:
        TIMING['ball_model_load'] = 0.0
    
    # Determine which client to return based on model URL
    client_key = 'player_detector' if "player" in model_url else 'ball_detector'
    
    # Check if we have a pre-initialized client
    with triton_client_lock:
        client = TRITON_CLIENTS.get(client_key)
        
        # If client exists, return it
        if client is not None:
            return client
        
        # Otherwise, attempt to initialize on demand as fallback
        logger.warning(f"Pre-initialized Triton client for {client_key} not available, creating new connection")
        
        try:
            # Extract server URL from model URL
            server_url = model_url.split("/")[2]  # Get host:port from http://host:port/model
            
            start_time = time.time()
            
            # Create the client with appropriate timeouts
            client = httpclient.InferenceServerClient(
                url=server_url,
                verbose=False,
                ssl=False,
            )
            
            TRITON_CLIENTS[client_key] = client
            load_time = time.time() - start_time
            
            # Update timing metrics
            if "player" in model_url:
                TIMING['player_model_load'] += load_time
                with model_cache_lock:
                    MODEL_LOADED_CACHE['player_detector'] = True
            elif "ball" in model_url:
                TIMING['ball_model_load'] += load_time
                with model_cache_lock:
                    MODEL_LOADED_CACHE['ball_detector'] = True
                
            logger.info(f"Triton client connected to {server_url} in {load_time:.2f}s (on-demand fallback)")
            return client
        except Exception as e:
            logger.error(f"Failed to create Triton client for {server_url}: {e}")
            return None


def check_models_ready(config: Dict[str, Any]) -> bool:
    """
    Check if Triton models are available on both servers.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if models are ready, False otherwise
    """
    global TRITON_CLIENTS
    
    with triton_client_lock:
        # If clients are already initialized, check their status
        player_client = TRITON_CLIENTS.get('player_detector')
        ball_client = TRITON_CLIENTS.get('ball_detector')
        
        if player_client is not None and ball_client is not None:
            try:
                # Extract player model name from URL
                player_model_url_parts = config['player_model_url'].split("/")
                player_model_name = player_model_url_parts[-1]
                
                # Extract ball model name from URL
                ball_model_url_parts = config['ball_model_url'].split("/")
                ball_model_name = ball_model_url_parts[-1]
                
                # Check if both servers are live and ready
                if (player_client.is_server_live() and 
                    player_client.is_server_ready() and 
                    player_client.is_model_ready(player_model_name) and
                    ball_client.is_server_live() and 
                    ball_client.is_server_ready() and 
                    ball_client.is_model_ready(ball_model_name)):
                    return True
                
                logger.warning("Pre-initialized Triton clients are not ready, will attempt to reinitialize")
            except Exception as e:
                logger.error(f"Error checking pre-initialized Triton clients: {e}")
    
    # If we don't have initialized clients or they're not ready, try initializing them
    return initialize_triton_clients(config)


def perform_yolo_inference_batched(
    model_url: str, 
    input_images: List[np.ndarray],
    conf_threshold: float,
    config: Dict[str, Any]
) -> List[Optional[sv.Detections]]:
    """
    Perform batched inference using Triton client API directly.
    
    Args:
        model_url: URL to the Triton server model endpoint
        input_images: List of input images as numpy arrays
        conf_threshold: Confidence threshold for detections
        config: Configuration dictionary
        
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
    if 'image_preprocessing' not in TIMING:
        TIMING['image_preprocessing'] = 0.0
    if 'data_conversion' not in TIMING:
        TIMING['data_conversion'] = 0.0
    if 'result_postprocessing' not in TIMING:
        TIMING['result_postprocessing'] = 0.0
    if 'detection_counts' not in TIMING:
        TIMING['detection_counts'] = 0
    
    try:
        # Skip empty batches
        if not input_images:
            return []
            
        # Get pre-initialized Triton client from global pool
        client = get_triton_client(model_url, config)
        
        if client is None:
            logger.error(f"Triton client not available for URL {model_url}")
            return [None] * len(input_images)
        
        # Extract model name from URL
        model_name = model_url.split("/")[-1]
        
        # Get model input size from config
        model_size = config['player_model_size'] if "player" in model_url else config['ball_model_size']
        input_width, input_height = model_size
        
        # Create batch of preprocessed images
        batch_size = len(input_images)
        logger.debug(f"Processing batch of size {batch_size} with model {model_name}")
        
        # ------------------------ GPU-ACCELERATED PREPROCESSING WITH PYTORCH -----------------------
        preprocess_start_time = time.time()
        
        # Store original shapes for scaling back
        original_shapes = [(img.shape[0], img.shape[1]) for img in input_images]
        
        # Use PyTorch for GPU acceleration if available
        if HAS_TORCH:
            # Initialize CUDA for processing
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.debug("Using CUDA for image preprocessing")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, falling back to CPU for preprocessing")
            
            # Define optimized preprocessing function
            def preprocess_batch_on_gpu(images):
                """Process entire batch of images on GPU at once."""
                # Check if images list is empty
                if not images:
                    return np.zeros((0, 3, input_height, input_width), dtype=np.float32)
                
                # Determine batch size
                batch_size = len(images)
                
                # Create a tensor to hold all images
                batch_tensors = torch.zeros((batch_size, 3, input_height, input_width), device=device)
                
                # Process the images in smaller sub-batches to avoid GPU memory issues
                sub_batch_size = 20  # Adjust based on GPU memory
                
                for start_idx in range(0, batch_size, sub_batch_size):
                    end_idx = min(start_idx + sub_batch_size, batch_size)
                    sub_batch = images[start_idx:end_idx]
                    
                    # Use vectorized operations for BGR to RGB conversion
                    # Allocate memory once for the entire sub-batch
                    rgb_batch = np.zeros((len(sub_batch), *sub_batch[0].shape), dtype=np.uint8)
                    
                    # Convert all images from BGR to RGB at once
                    for i, img in enumerate(sub_batch):
                        rgb_batch[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Convert the entire sub-batch to tensor at once
                    # This avoids multiple transfers between CPU and GPU
                    sub_batch_tensor = torch.from_numpy(rgb_batch).to(device).float()
                    
                    # Rearrange from NHWC to NCHW format - batch transpose is faster
                    sub_batch_tensor = sub_batch_tensor.permute(0, 3, 1, 2)
                    
                    # Resize all images in one operation
                    sub_batch_tensor = F.resize(sub_batch_tensor, [input_height, input_width])
                    
                    # Normalize in one operation (0-255 to 0-1)
                    sub_batch_tensor = sub_batch_tensor / 255.0
                    
                    # Add to the main batch tensor
                    batch_tensors[start_idx:end_idx] = sub_batch_tensor
                
                # Return as numpy array in NCHW format
                return batch_tensors.cpu().numpy()
            
            # Process batch using GPU acceleration
            batch_data = preprocess_batch_on_gpu(input_images)
        else:
            # Fallback to CPU processing if PyTorch not available
            logger.warning("PyTorch not available, using CPU for preprocessing")
            
            # Create a batch with the right dimensions
            batch_data = np.zeros((batch_size, 3, input_height, input_width), dtype=np.float32)
            
            # Process each image
            for i, img in enumerate(input_images):
                # Convert from BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize the image
                img_resized = cv2.resize(img_rgb, (input_width, input_height))
                
                # Convert to CHW format and normalize
                img_chw = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                
                # Add to batch
                batch_data[i] = img_chw
        
        # End timing for preprocessing
        preprocess_time = time.time() - preprocess_start_time
        TIMING['image_preprocessing'] += preprocess_time
        logger.debug(f"Preprocessing completed in {preprocess_time:.3f}s")
        
        # Start timing for data conversion
        data_conversion_start = time.time()
        
        # Prepare inputs for Triton
        inputs = [httpclient.InferInput("images", batch_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch_data)
        
        # Request outputs
        outputs = [httpclient.InferRequestedOutput("output0")]
        
        # End timing for data conversion
        data_conversion_time = time.time() - data_conversion_start
        TIMING['data_conversion'] += data_conversion_time
        
        # Start timing for pure inference
        pure_inference_start = time.time()
        
        # Perform inference
        results = client.infer(model_name, inputs, outputs=outputs)
        
        # Get pure inference time
        pure_inference_time = time.time() - pure_inference_start
        
        # Update pure inference timing metrics based on model type
        if "player" in model_url:
            TIMING['player_pure_inference'] += pure_inference_time
            logger.debug(f"Pure player inference time for batch of {batch_size} images: {pure_inference_time:.3f}s")
        elif "ball" in model_url:
            TIMING['ball_pure_inference'] += pure_inference_time
            logger.debug(f"Pure ball inference time for batch of {batch_size} images: {pure_inference_time:.3f}s")
        
        # ------------------------ OPTIMIZED POSTPROCESSING -----------------------
        postprocess_start_time = time.time()
        
        # Get output data
        output_data = results.as_numpy("output0")
        
        # Create list to store detections for each image
        all_detections = []
        
        # Define postprocessing function for parallel execution
        def process_image_detections(idx):
            # Get original dimensions for scaling
            orig_h, orig_w = original_shapes[idx]
            scale_x = orig_w / input_width
            scale_y = orig_h / input_height
            
            # Get the output for this image
            image_output = output_data[idx]  # Shape: [5, num_detections]
            
            # Find detections that meet confidence threshold
            valid_detections = image_output[4, :] > conf_threshold
            
            # Skip further processing if no valid detections
            if not np.any(valid_detections):
                return sv.Detections(
                    xyxy=np.zeros((0, 4)),
                    confidence=np.array([]),
                    class_id=np.array([]),
                )
            
            # Extract data for valid detections only
            cx = image_output[0, valid_detections]  # center x
            cy = image_output[1, valid_detections]  # center y
            w = image_output[2, valid_detections]   # width
            h = image_output[3, valid_detections]   # height
            confidences = image_output[4, valid_detections]  # confidence scores
            
            # Check if coordinates are normalized (between 0 and 1)
            is_normalized = np.all((0 <= cx) & (cx <= 1) & (0 <= cy) & (cy <= 1) & 
                                   (0 <= w) & (w <= 1) & (0 <= h) & (h <= 1))
            
            # Compute all coordinates in a vectorized way
            if is_normalized:
                # For normalized coordinates
                x1 = (cx - w/2) * input_width * scale_x
                y1 = (cy - h/2) * input_height * scale_y
                x2 = (cx + w/2) * input_width * scale_x
                y2 = (cy + h/2) * input_height * scale_y
            else:
                # For pixel coordinates
                x1 = (cx - w/2) * scale_x
                y1 = (cy - h/2) * scale_y
                x2 = (cx + w/2) * scale_x
                y2 = (cy + h/2) * scale_y
            
            # Clip to image boundaries in a vectorized way
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            # Find valid boxes (where x2 > x1 and y2 > y1)
            valid_boxes = (x2 > x1) & (y2 > y1)
            
            # Skip if no valid boxes
            if not np.any(valid_boxes):
                return sv.Detections(
                    xyxy=np.zeros((0, 4)),
                    confidence=np.array([]),
                    class_id=np.array([]),
                )
            
            # Create the boxes array
            boxes = np.column_stack([
                x1[valid_boxes], 
                y1[valid_boxes], 
                x2[valid_boxes], 
                y2[valid_boxes]
            ])
            
            # Create detection object
            return sv.Detections(
                xyxy=boxes,
                confidence=confidences[valid_boxes],
                class_id=np.zeros(len(boxes), dtype=int),  # Default to class 0
            )
        
        # Process each image's detections
        # Use parallel processing for larger batches
        max_workers = config.get('max_workers', 4)
        if batch_size >= 4 and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, batch_size)) as executor:
                all_detections = list(executor.map(process_image_detections, range(batch_size)))
        else:
            # For small batches, sequential is faster due to thread overhead
            all_detections = [process_image_detections(i) for i in range(batch_size)]
        
        # Log detection counts
        for i, detections in enumerate(all_detections):
            if detections.xyxy.shape[0] > 0:
                logger.debug(f"Batch item {i} detected {detections.xyxy.shape[0]} objects with confidence >= {conf_threshold}")
            else:
                logger.debug(f"Batch item {i} detected no objects")
        
        # End timing for postprocessing
        postprocess_time = time.time() - postprocess_start_time
        TIMING['result_postprocessing'] += postprocess_time
        
        # Update total timing metrics based on model type
        # This total includes preprocessing + inference + postprocessing
        total_inference_time = preprocess_time + data_conversion_time + pure_inference_time + postprocess_time
        if "player" in model_url:
            TIMING['player_detection'] += total_inference_time
            logger.debug(f"Batched player detection ({batch_size} images) completed in {total_inference_time:.3f}s using {model_url}")
        elif "ball" in model_url:
            TIMING['ball_detection'] += total_inference_time
            logger.debug(f"Batched ball detection ({batch_size} images) completed in {total_inference_time:.3f}s using {model_url}")
        
        TIMING['detection_counts'] += batch_size
        
        return all_detections
        
    except Exception as e:
        logger.error(f"Error performing batched Triton inference with model at {model_url}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [None] * len(input_images)


def warmup_triton_inference(config: Dict[str, Any]) -> bool:
    """
    Perform warm-up inferences to initialize Triton client connections.
    This ensures that the first actual inference doesn't include connection establishment time.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    global TRITON_CLIENTS
    
    try:
        logger.info("Warming up Triton inference connections...")
        start_time = time.time()
        
        # Create a small dummy batch for warm-up
        dummy_batch = np.zeros((20, 3, 640, 640), dtype=np.float32)
        
        # Warm up player detector
        player_client = TRITON_CLIENTS.get('player_detector')
        ball_client = TRITON_CLIENTS.get('ball_detector')
        
        if player_client:
            try:
                # Extract player model name from URL
                player_model_url_parts = config['player_model_url'].split("/")
                player_model_name = player_model_url_parts[-1]
                
                # Create dummy input
                inputs = [httpclient.InferInput("images", dummy_batch.shape, "FP32")]
                inputs[0].set_data_from_numpy(dummy_batch)
                
                # Request outputs
                outputs = [httpclient.InferRequestedOutput("output0")]
                
                # Perform warm-up inference
                logger.info("Warming up player detector...")
                player_client.infer(player_model_name, inputs, outputs=outputs)
                logger.info("Player detector warm-up complete")
            except Exception as e:
                logger.warning(f"Error warming up player detector: {e}")
        
        if ball_client:
            try:
                # Extract ball model name from URL
                ball_model_url_parts = config['ball_model_url'].split("/")
                ball_model_name = ball_model_url_parts[-1]
                
                # Create dummy input
                inputs = [httpclient.InferInput("images", dummy_batch.shape, "FP32")]
                inputs[0].set_data_from_numpy(dummy_batch)
                
                # Request outputs
                outputs = [httpclient.InferRequestedOutput("output0")]
                
                # Perform warm-up inference
                logger.info("Warming up ball detector...")
                ball_client.infer(ball_model_name, inputs, outputs=outputs)
                logger.info("Ball detector warm-up complete")
            except Exception as e:
                logger.warning(f"Error warming up ball detector: {e}")
        
        warmup_time = time.time() - start_time
        logger.info(f"Triton inference warm-up completed in {warmup_time:.3f}s")
        return True
    except Exception as e:
        logger.warning(f"Error during Triton inference warm-up: {e}")
        return False
