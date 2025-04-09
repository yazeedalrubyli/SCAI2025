"""
Thread pool utilities for Soccer3D.

This module provides thread pool management to reduce overhead from creating
and destroying thread pools repeatedly.
"""
import os
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("Soccer3D")

# Detect available CPU cores
CPU_COUNT = multiprocessing.cpu_count()
logger.info(f"Detected {CPU_COUNT} CPU cores")

# Default worker counts based on task type
DEFAULT_IO_WORKERS = min(32, CPU_COUNT * 2)  # More workers for I/O bound tasks
DEFAULT_CPU_WORKERS = max(8, CPU_COUNT - 2)  # Slightly fewer than total cores for CPU-bound tasks
DEFAULT_BALANCED_WORKERS = max(12, CPU_COUNT)  # Balance between I/O and CPU

# Global thread pools for different operations
_THREAD_POOLS = {}
_THREAD_POOLS_LOCK = threading.RLock()

# Default worker counts for each pool type
POOL_CONFIGS = {
    # I/O-bound pools
    'image_loading': DEFAULT_IO_WORKERS,
    'frame_loading': DEFAULT_IO_WORKERS,
    
    # CPU-bound pools (computation intensive)
    'batch_processing': DEFAULT_CPU_WORKERS,
    'ray_processing': DEFAULT_CPU_WORKERS,
    'keypoint_triangulation': DEFAULT_CPU_WORKERS,
    
    # Balanced pools
    'pose_processing': DEFAULT_BALANCED_WORKERS,
    
    # Small, specialized pools
    'triangulation': 3  # Only need 3 workers for the 3 triangulation tasks
}

def get_thread_pool(pool_name, max_workers=None):
    """
    Get or create a thread pool with the specified name.
    
    Args:
        pool_name: Name of the thread pool
        max_workers: Maximum number of workers (if None, uses optimized defaults)
        
    Returns:
        ThreadPoolExecutor instance
    """
    global _THREAD_POOLS, _THREAD_POOLS_LOCK
    
    # If max_workers is not specified, use our optimized defaults
    if max_workers is None:
        if pool_name in POOL_CONFIGS:
            max_workers = POOL_CONFIGS[pool_name]
        else:
            # Default to balanced workers if pool type is unknown
            max_workers = DEFAULT_BALANCED_WORKERS
    
    with _THREAD_POOLS_LOCK:
        if pool_name not in _THREAD_POOLS:
            logger.debug(f"Creating thread pool '{pool_name}' with {max_workers} workers")
            _THREAD_POOLS[pool_name] = ThreadPoolExecutor(max_workers=max_workers)
        
        return _THREAD_POOLS[pool_name]

def shutdown_thread_pools():
    """
    Shutdown all thread pools cleanly.
    Call this before application exit.
    """
    global _THREAD_POOLS, _THREAD_POOLS_LOCK
    
    with _THREAD_POOLS_LOCK:
        for name, pool in _THREAD_POOLS.items():
            logger.debug(f"Shutting down thread pool '{name}'")
            pool.shutdown(wait=True)
        
        _THREAD_POOLS.clear() 