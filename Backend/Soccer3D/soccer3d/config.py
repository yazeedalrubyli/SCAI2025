"""
Configuration module for Soccer3D.
"""
import os
import yaml
import multiprocessing
from typing import Dict, Any, Optional

# Detect CPU core count for optimal performance
CPU_COUNT = multiprocessing.cpu_count()

# Default configuration parameters
DEFAULT_CONFIG = {
    'player_conf_threshold': 0.5,
    'ball_conf_threshold': 0.1,
    'pose_visibility_threshold': 0.5,
    'pose_from_detection_padding': 20,
    'max_workers': max(16, CPU_COUNT - 2),  # Use most cores but leave some for system
    'player_model_path': "soccer3d/models/player_model/model.engine", # Path to player detection model
    'ball_model_path': "soccer3d/models/ball_model/model.engine",     # Path to ball detection model
    'player_model_size': (640, 640),
    'ball_model_size': (640, 640),
    'silence_model_output': True,
    'field_orientation': {
        'side_axis': 'x',
        'goal_axis': 'y',
        'up_axis': 'z'
    },
    'max_ray_cache_size': 10000,
    'mp_pose_pool_size': max(8, min(CPU_COUNT // 2, 20)),  # Half of cores, max 20
    'mp_pose_complexity': 0,
    'log_level': 'INFO',
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with defaults.
    
    Args:
        config_path: Path to a YAML configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            if user_config:
                # Merge user configuration with defaults
                _merge_configs(config, user_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration")
    
    return config


def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
    """
    Recursively merge override_config into base_config.
    
    Args:
        base_config: Base configuration to update
        override_config: Configuration with values to override
    """
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")