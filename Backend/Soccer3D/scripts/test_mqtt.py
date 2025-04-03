#!/usr/bin/env python
"""
Test MQTT publishing from Soccer3D.

This script creates a simple test JSON payload and publishes it to the MQTT broker.
It can be used to verify that the MQTT broker is accessible and working correctly.
"""
import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

# Add parent directory to path to allow running script from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import MQTT client
try:
    import paho.mqtt.client as mqtt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paho-mqtt"])
    import paho.mqtt.client as mqtt

# Import Soccer3D modules
from soccer3d import initialize

# Set up logger
logger = logging.getLogger("Soccer3D_MQTT_Test")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test MQTT publishing from Soccer3D")
    
    # MQTT options
    parser.add_argument("--mqtt-broker", type=str, default="localhost",
                        help="MQTT broker address (default: localhost)")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port (default: 1883)")
    parser.add_argument("--mqtt-topic", type=str, default="soccer3d/test",
                        help="MQTT topic for test data (default: soccer3d/test)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--test-duration", type=float, default=5.0,
                        help="Duration in seconds to run the performance test (default: 5.0)")
    parser.add_argument("--message-size", type=str, choices=["small", "medium", "large"], default="small",
                        help="Size of test messages to send (default: small)")
    
    return parser.parse_args()


def create_test_payload(size="small", message_id=0):
    """
    Create a test payload with variable size.
    
    Args:
        size: Size of the payload ("small", "medium", "large")
        message_id: Unique identifier for the message
        
    Returns:
        Dictionary with test data
    """
    # Base payload
    payload = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "type": "Soccer3D MQTT Test",
            "message_id": message_id
        },
        "test_data": {
            "player": {
                "position": {
                    "x": 10.5,
                    "y": 20.3,
                    "z": 0.0
                }
            },
            "ball": {
                "position": {
                    "x": 12.1,
                    "y": 22.7,
                    "z": 0.5
                }
            }
        }
    }
    
    # Add more data for medium and large payloads
    if size == "medium" or size == "large":
        # Add player pose data (33 keypoints)
        payload["test_data"]["player"]["pose_keypoints"] = {}
        for i in range(33):
            payload["test_data"]["player"]["pose_keypoints"][f"keypoint_{i}"] = {
                "x": 10.0 + i * 0.1,
                "y": 20.0 + i * 0.2,
                "z": 0.5 + i * 0.01
            }
    
    if size == "large":
        # Add additional players (10 more)
        payload["test_data"]["additional_players"] = []
        for i in range(30):
            player = {
                "id": i + 1,
                "position": {
                    "x": 15.0 + i * 1.0,
                    "y": 25.0 + i * 1.5,
                    "z": 0.0
                },
                "pose_keypoints": {}
            }
            
            # Add keypoints for each additional player
            for j in range(33):
                player["pose_keypoints"][f"keypoint_{j}"] = {
                    "x": 15.0 + i * 1.0 + j * 0.1,
                    "y": 25.0 + i * 1.5 + j * 0.2,
                    "z": 0.0 + j * 0.01
                }
            
            payload["test_data"]["additional_players"].append(player)
    
    return payload


def perform_speed_test(client, topic, duration=5.0, message_size="small"):
    """
    Perform a speed test by sending messages continuously for the specified duration.
    
    Args:
        client: MQTT client instance
        topic: Topic to publish to
        duration: Duration in seconds to run the test
        message_size: Size of messages to send
        
    Returns:
        Tuple of (messages_sent, messages_per_second)
    """
    logger.info(f"Starting MQTT speed test for {duration} seconds with {message_size} messages...")
    
    # Initialize counters
    messages_sent = 0
    start_time = time.time()
    end_time = start_time + duration
    
    # Send messages until time is up
    while time.time() < end_time:
        # Create payload with unique message ID
        payload = create_test_payload(size=message_size, message_id=messages_sent)
        json_str = json.dumps(payload)
        
        # Publish message
        result = client.publish(topic, json_str)
        
        # Check if successful
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            messages_sent += 1
        
        # Optional: add a small delay to prevent overwhelming the broker
        # time.sleep(0.001)
    
    # Calculate actual duration and messages per second
    actual_duration = time.time() - start_time
    messages_per_second = messages_sent / actual_duration
    
    logger.info(f"Speed test completed in {actual_duration:.2f} seconds")
    logger.info(f"Sent {messages_sent} messages ({message_size})")
    logger.info(f"Average speed: {messages_per_second:.2f} messages per second")
    
    return messages_sent, messages_per_second


def main():
    """Main entry point for the test script."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    # Create a sample payload to show the structure
    sample_payload = create_test_payload(size=args.message_size)
    json_str = json.dumps(sample_payload, indent=2)
    
    # Log payload example
    logger.info(f"Example payload ({args.message_size} size):")
    if args.log_level == "DEBUG":
        logger.debug(json_str)
    else:
        # Show abbreviated payload info in non-debug mode
        logger.info(f"Payload size: {len(json_str)} bytes")
    
    # Publish to MQTT broker
    try:
        broker = args.mqtt_broker
        port = args.mqtt_port
        topic = args.mqtt_topic
        
        # Create MQTT client
        client = mqtt.Client()
        
        # Connect to broker
        logger.info(f"Connecting to MQTT broker at {broker}:{port}")
        client.connect(broker, port, 60)
        
        # Perform speed test
        messages_sent, messages_per_second = perform_speed_test(
            client, 
            topic, 
            duration=args.test_duration,
            message_size=args.message_size
        )
        
        # Disconnect
        client.disconnect()
        
        # Print summary
        logger.info("=== MQTT Performance Test Results ===")
        logger.info(f"Test duration: {args.test_duration} seconds")
        logger.info(f"Message size: {args.message_size}")
        logger.info(f"Messages sent: {messages_sent}")
        logger.info(f"Average speed: {messages_per_second:.2f} messages/second")
        
    except Exception as e:
        logger.error(f"Error during MQTT test: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())