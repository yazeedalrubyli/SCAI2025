# Soccer3D

Soccer player and ball detection with 3D pose estimation using Ultralytics YOLO and MediaPipe.

## Features

- Real-time soccer player detection using Ultralytics YOLO models
- Ball tracking with configurable confidence thresholds
- 3D pose estimation using MediaPipe
- 3D position triangulation from multiple camera views
- Player orientation estimation and cardinal direction mapping
- JSON output with detailed 3D information
- Batch processing capabilities for improved performance
- Optional MQTT integration for real-time data publishing

## Requirements

- Python 3.7+
- NVIDIA GPU with CUDA support (recommended)
- Ultralytics YOLO for object detection
- MediaPipe for pose estimation
- MQTT broker (optional, for real-time data publishing)

## Installation

```bash
# Install from source
git clone https://github.com/yazeedalrubyli/soccer3d.git
cd soccer3d
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic usage

```bash
# Process a specific frame
python -m soccer3d.scripts.run_soccer3d --frame 160
```

### Advanced options

```bash
# Specify custom configuration
python -m soccer3d.scripts.run_soccer3d --config path/to/config.yaml

# Process a range of frames
python -m soccer3d.scripts.run_soccer3d --start-frame 150 --end-frame 200

# Batch processing with performance options
python -m soccer3d.scripts.run_soccer3d --start-frame 150 --end-frame 200 --batch-size 20 --max-workers 8 --preload-all

# Publish data to MQTT broker
python -m soccer3d.scripts.run_soccer3d --frame 160 --mqtt-broker localhost --mqtt-port 1883 --mqtt-topic soccer3d/data
```

### Performance Optimization

Soccer3D offers several performance optimization options:

```bash
# Enable frame preloading for faster processing
python -m soccer3d.scripts.run_soccer3d --frame 160 --preload-all

# Configure batch processing size
python -m soccer3d.scripts.run_soccer3d --frame 160 --batch-size 20

# Control thread worker count
python -m soccer3d.scripts.run_soccer3d --frame 160 --max-workers 16
```

### MQTT Integration

Soccer3D can publish JSON results to an MQTT broker in real-time:

```bash
# Test MQTT connectivity
python -m soccer3d.scripts.test_mqtt --mqtt-broker localhost --mqtt-port 1883

# Process frame and publish to MQTT
python -m soccer3d.scripts.run_soccer3d --frame 160 --mqtt-broker localhost
```

MQTT options:
- `--mqtt-broker`: MQTT broker address (default: localhost)
- `--mqtt-port`: MQTT broker port (default: 1883)
- `--mqtt-topic`: MQTT topic for publishing data (default: soccer3d/data)
- `--mqtt-disable`: Disable MQTT publishing

## Configuration

The default configuration parameters can be found in `soccer3d/config.py`. You can override these by providing a custom YAML configuration file:

```bash
python -m soccer3d.scripts.run_soccer3d --config my_config.yaml
```

Key configuration parameters include:
- Detection thresholds for players and balls
- Model paths and parameters
- Performance settings (workers, batch sizes)
- Pose visibility thresholds
- Field orientation mapping

## Project Structure

```
soccer3d/
├── soccer3d/            # Main package
│   ├── models/          # Detection and pose models
│   ├── utils/           # Utility functions including camera handling
│   ├── data/            # Data handling
│   ├── scripts/         # Command line tools
│   │   ├── run_soccer3d.py  # Main processing script
│   │   └── test_mqtt.py     # MQTT testing utility
│   ├── config.py        # Configuration management
│   └── logger.py        # Logging system
├── data/                # Data directory for input images
├── requirements.txt     # Python dependencies
└── setup.py             # Installation script
```

## License

MIT
