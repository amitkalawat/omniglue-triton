#!/bin/bash
# Script to deploy OmniGlue in NVIDIA Triton Inference Server
# This script sets up OmniGlue, SuperPoint, and DINOv2 as an ensemble in Triton

set -e  # Exit on any error

# Create directories for the project
echo "Creating project directories..."
mkdir -p omniglue-triton
cd omniglue-triton

# Create conda environment (if conda is installed)
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda create -n omniglue-triton python=3.8 -y
    conda activate omniglue-triton
else
    echo "Conda not found. Please install conda or set up a virtual environment manually."
    echo "Continuing with system Python..."
fi

# Install dependencies
echo "Installing dependencies..."
pip install numpy opencv-python onnx onnxruntime pillow tritonclient torch torchvision

# Create model repository structure
echo "Creating model repository structure..."
mkdir -p model_repository/{superpoint,dinov2,omniglue,omniglue_ensemble}/{1,}

# Download the models
echo "Downloading models..."
mkdir -p models
cd models

# Download SuperPoint
echo "Downloading SuperPoint..."
if [ ! -d "SuperPoint" ]; then
    git clone https://github.com/rpautrat/SuperPoint.git
    mv SuperPoint/pretrained_models/sp_v6.tgz .
    rm -rf SuperPoint
    tar zxvf sp_v6.tgz
    rm sp_v6.tgz
fi

# Download DINOv2
echo "Downloading DINOv2..."
if [ ! -f "dinov2_vitb14_pretrain.pth" ]; then
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
fi

# Download OmniGlue
echo "Downloading OmniGlue..."
if [ ! -d "og_export" ]; then
    wget https://storage.googleapis.com/omniglue/og_export.zip
    unzip og_export.zip
    rm og_export.zip
fi

cd ..

# Convert models to ONNX format
echo "Converting models to ONNX format..."
python convert_models_to_onnx.py

# Create configuration files
echo "Creating configuration files..."

# SuperPoint configuration
cat > model_repository/superpoint/config.pbtxt << EOL
name: "superpoint"
platform: "onnxruntime_onnx"
max_batch_size: 2
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 1, 480, 640 ]
  }
]
output [
  {
    name: "keypoints"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "descriptors"
    data_type: TYPE_FP32
    dims: [ -1, 256 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
EOL

# DINOv2 configuration
cat > model_repository/dinov2/config.pbtxt << EOL
name: "dinov2"
platform: "onnxruntime_onnx"
max_batch_size: 2
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
EOL

# OmniGlue configuration
cat > model_repository/omniglue/config.pbtxt << EOL
name: "omniglue"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {
    name: "keypoints0"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "descriptors0"
    data_type: TYPE_FP32
    dims: [ -1, 256 ]
  },
  {
    name: "keypoints1"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "descriptors1"
    data_type: TYPE_FP32
    dims: [ -1, 256 ]
  },
  {
    name: "features0"
    data_type: TYPE_FP32
    dims: [ 768 ]
  },
  {
    name: "features1"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
output [
  {
    name: "matches_kp0"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "matches_kp1"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "match_confidences"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
EOL

# Ensemble configuration
cat > model_repository/omniglue_ensemble/config.pbtxt << EOL
name: "omniglue_ensemble"
platform: "ensemble"
max_batch_size: 0
input [
  {
    name: "image0"
    data_type: TYPE_FP32
    dims: [ 3, 480, 640 ]
  },
  {
    name: "image1"
    data_type: TYPE_FP32
    dims: [ 3, 480, 640 ]
  }
]
output [
  {
    name: "matches_kp0"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "matches_kp1"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  },
  {
    name: "match_confidences"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "superpoint"
      model_version: -1
      input_map {
        key: "image"
        value: "image0"
      }
      output_map {
        key: "keypoints"
        value: "keypoints0"
      }
      output_map {
        key: "descriptors"
        value: "descriptors0"
      }
    },
    {
      model_name: "superpoint"
      model_version: -1
      input_map {
        key: "image"
        value: "image1"
      }
      output_map {
        key: "keypoints"
        value: "keypoints1"
      }
      output_map {
        key: "descriptors"
        value: "descriptors1"
      }
    },
    {
      model_name: "dinov2"
      model_version: -1
      input_map {
        key: "image"
        value: "image0"
      }
      output_map {
        key: "features"
        value: "features0"
      }
    },
    {
      model_name: "dinov2"
      model_version: -1
      input_map {
        key: "image"
        value: "image1"
      }
      output_map {
        key: "features"
        value: "features1"
      }
    },
    {
      model_name: "omniglue"
      model_version: -1
      input_map {
        key: "keypoints0"
        value: "keypoints0"
      }
      input_map {
        key: "descriptors0"
        value: "descriptors0"
      }
      input_map {
        key: "keypoints1"
        value: "keypoints1"
      }
      input_map {
        key: "descriptors1"
        value: "descriptors1"
      }
      input_map {
        key: "features0"
        value: "features0"
      }
      input_map {
        key: "features1"
        value: "features1"
      }
      output_map {
        key: "matches_kp0"
        value: "matches_kp0"
      }
      output_map {
        key: "matches_kp1"
        value: "matches_kp1"
      }
      output_map {
        key: "match_confidences"
        value: "match_confidences"
      }
    }
  ]
}
EOL

echo "Configuration files created."

# Create a script to start the Triton server
cat > start_triton.sh << EOL
#!/bin/bash
# Start Triton Inference Server with the OmniGlue ensemble

docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v \${PWD}/model_repository:/models \\
  nvcr.io/nvidia/tritonserver:23.04-py3 \\
  tritonserver --model-repository=/models --log-verbose=1
EOL

chmod +x start_triton.sh

echo "Created start_triton.sh script."

echo "Setup complete! Follow these steps to run OmniGlue in Triton:"
echo "1. Complete the model conversion with 'python convert_models_to_onnx.py'"
echo "2. Start Triton with './start_triton.sh'"
echo "3. Test the deployment with 'python test_client.py'"
echo ""
echo "To push this to GitHub, use:"
echo "git init"
echo "git add ."
echo "git commit -m 'Initial commit'"
echo "git branch -M main"
echo "git remote add origin https://github.com/YOUR-USERNAME/omniglue-triton.git"
echo "git push -u origin main"
