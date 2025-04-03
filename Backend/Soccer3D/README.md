# Soccer3D

Soccer player and ball detection with 3D pose estimation using MediaPipe and Triton Inference Server.

## Features

- Real-time soccer player detection
- Ball tracking
- 3D pose estimation using MediaPipe
- 3D position triangulation from multiple camera views
- Player orientation estimation
- JSON output with detailed 3D information
- MQTT integration for real-time data publishing

## Requirements

- Python 3.7+
- NVIDIA GPU with CUDA support (recommended)
- Triton Inference Server running with YOLOv12 models
- MQTT broker (optional, for real-time data publishing)

## Installation

```bash
# Install from source
git clone https://github.com/yazeedalrubyli/soccer3d.git
cd soccer3d
pip install -e .
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

# Publish data to MQTT broker
python -m soccer3d.scripts.run_soccer3d --frame 160 --mqtt-broker localhost --mqtt-port 1883 --mqtt-topic soccer3d/data
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

Edit `config.yaml` to customize:

- Detection thresholds
- Triton server URLs
- Camera parameters
- Output formats

## Project Structure

```
soccer3d/
├── soccer3d/            # Main package
│   ├── models/          # Detection and pose models
│   ├── utils/           # Utility functions
│   └── data/            # Data handling
├── scripts/             # Command line tools
│   ├── run_soccer3d.py  # Main processing script
│   └── test_mqtt.py     # MQTT testing utility
├── tests/               # Unit tests
├── logs/                # Log output
└── setup.py             # Installation script
```

## License

MIT
