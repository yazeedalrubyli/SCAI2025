"""
Soccer3D: Soccer Player and Ball Detection with 3D Pose Estimation
"""
import atexit

__version__ = "0.1.0"

from soccer3d.config import load_config
from soccer3d.logger import setup_logger, SuppressOutput
from soccer3d.utils.threading import shutdown_thread_pools

# Global variables
CONFIG = None
logger = None

# Register thread pool cleanup at exit
atexit.register(shutdown_thread_pools)

# Initialize configuration and logger
def initialize(config_path=None):
    """
    Initialize the Soccer3D package.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        Tuple of (CONFIG, logger)
    """
    global CONFIG, logger
    
    # Load configuration
    CONFIG = load_config(config_path)
    
    # Setup logger
    logger = setup_logger(CONFIG)
    
    return CONFIG, logger