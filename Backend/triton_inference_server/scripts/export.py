from ultralytics import YOLO
from pathlib import Path
import contextlib
import subprocess
import time
import shutil

from tritonclient.http import InferenceServerClient

def export_model(model_path, model_name, triton_model_path):
    # Load a model
    model = YOLO(model_path)

    # Retrieve metadata during export
    metadata = []

    def export_cb(exporter):
        metadata.append(exporter.metadata)

    model.add_callback("on_export_end", export_cb)

    # Export the model
    onnx_file = model.export(format="onnx", dynamic=True, batch=20, device=0)

    # Create directories
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)

    # Move ONNX model to Triton Model path
    Path(onnx_file).replace(triton_model_path / "1" / "model.onnx")

    # Create config file
    (triton_model_path / "config.pbtxt").touch()

    data = """
max_batch_size: 20

# Add metadata
parameters {
  key: "metadata"
  value {
    string_value: "%s"
  }
}

# Model warmup configuration
model_warmup {
  name: "warmup"
  batch_size: 20
  inputs {
    key: "images"
    value {
      data_type: TYPE_FP32
      dims: [3, 640, 640]
      zero_data: true
    }
  }
}

# (Optional) Enable TensorRT for GPU inference
# First run will be slow due to TensorRT engine conversion
optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      name: "tensorrt"
      parameters {
        key: "precision_mode"
        value: "FP16"
      }
      parameters {
        key: "trt_engine_cache_enable"
        value: "1"
      }
      parameters {
        key: "trt_engine_cache_path"
        value: "/models/%s/1"
      }
    }
  }
}
""" % (metadata[0], model_name)  # noqa

    with open(triton_model_path / "config.pbtxt", "w") as f:
        f.write(data)

# Define base path
base_path = Path.cwd()

# Create separate model repositories for each model
ball_repo_path = base_path / "ball_model"
player_repo_path = base_path / "player_model"

# Delete model folders if they exist
if ball_repo_path.exists():
    shutil.rmtree(ball_repo_path)
if player_repo_path.exists():
    shutil.rmtree(player_repo_path)

# Create directories
ball_repo_path.mkdir(parents=True, exist_ok=True)
player_repo_path.mkdir(parents=True, exist_ok=True)

# Export ball detector
ball_model_path = "models/ball/best.pt"
ball_model_name = "ball_detector"
export_model(ball_model_path, ball_model_name, ball_repo_path / ball_model_name)

# Export player detector
player_model_path = "models/player/best.pt"
player_model_name = "player_detector"
export_model(player_model_path, player_model_name, player_repo_path / player_model_name)

tag = "nvcr.io/nvidia/tritonserver:24.09-py3"

# Pull the image
subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server for ball detector on port 8000
ball_container_id = (
    subprocess.check_output(
        f"docker run -d --rm --gpus all -v {ball_repo_path.absolute()}:/models -p 8000:8001 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

# Run the Triton server for player detector on port 8001
player_container_id = (
    subprocess.check_output(
        f"docker run -d --rm --gpus all -v {player_repo_path.absolute()}:/models -p 8001:8001 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

print(f"Started ball detector server in container: {ball_container_id}")
print(f"Started player detector server in container: {player_container_id}")

# Wait for the Triton servers to start
ball_triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
player_triton_client = InferenceServerClient(url="localhost:8001", verbose=False, ssl=False)

# Wait until ball model is ready
print("Waiting for ball detector model to be ready...")
for _ in range(10):
    with contextlib.suppress(Exception):
        if ball_triton_client.is_model_ready(ball_model_name):
            print(f"Ball detector model is ready on port 8000")
            break
        print("Waiting for ball detector model...")
    time.sleep(1)

# Wait until player model is ready
print("Waiting for player detector model to be ready...")
for _ in range(10):
    with contextlib.suppress(Exception):
        if player_triton_client.is_model_ready(player_model_name):
            print(f"Player detector model is ready on port 8001")
            break
        print("Waiting for player detector model...")
    time.sleep(1)

print("\nAll servers are running:")
print(f"- Ball detector available at: http://localhost:8000/v2/models/{ball_model_name}")
print(f"- Player detector available at: http://localhost:8001/v2/models/{player_model_name}")
print("\nTo stop the servers run:")
print(f"docker stop {ball_container_id} {player_container_id}")