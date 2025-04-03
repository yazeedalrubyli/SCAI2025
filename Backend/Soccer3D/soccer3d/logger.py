"""
Logging module for Soccer3D.
"""
import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Configure and return the Soccer3D logger.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log filename
    log_filename = os.path.join(log_dir, f"soccer3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Get log level from config or default to INFO
    log_level_name = config.get('log_level', 'INFO') if config else 'INFO'
    log_level = getattr(logging, log_level_name)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create the logger
    logger = logging.getLogger("Soccer3D")
    
    # Silence other loggers
    for logger_name in ['PIL', 'matplotlib', 'ultralytics', 'torch', 'tensorflow']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    return logger


class SuppressOutput:
    """
    Context manager to suppress stdout/stderr during function calls.
    """
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
