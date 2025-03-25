#!/bin/bash
# Improved script to set up OmniGlue with Triton Python backend

set -e  # Exit on any error

echo "Setting up OmniGlue with Triton Python backend (Improved Version)"

# Create project directory
PROJECT_DIR="omniglue-triton-python"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create model repository structure
echo "Creating model repository structure..."
mkdir -p model_repository/omniglue_python/1/models

# Copy the Python model file
echo "Creating Python backend model file..."
cat > model_repository/omniglue_python/1/model.py << 'EOL'
# The model.py content will be copied here by the setup script
# This will be a placeholder until we copy the actual improved_omniglue_backend.py content
EOL

# Download model files (if not already present)
echo "Downloading or checking for model files..."

# Function to check and download model files
download_models() {
    cd model_repository/omniglue_python/1/models
    
    # Download SuperPoint if not exists
    if [ ! -d "sp_v6" ]; then
        echo "Downloading SuperPoint..."
        if [ -f "sp_v6.tgz" ]; then
            # Extract if already downloaded
            mkdir -p sp_v6
            tar -xzf sp_v6.tgz -C sp_v6
        else
            # Download and extract
            wget -q https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz
            mkdir -p sp_v6
            tar -xzf sp_v6.tgz -C sp_v6
            rm sp_v6.tgz
        fi
        echo "SuperPoint model downloaded and extracted."
    else
        echo "SuperPoint model already exists."
    fi
    
    # Download DINOv2 if not exists
    if [ ! -f "dinov2_vitb14_pretrain.pth" ]; then
        echo "Downloading DINOv2..."
        wget -q https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
        echo "DINOv2 model downloaded."
    else
        echo "DINOv2 model already exists."
    fi
    
    # Download OmniGlue if not exists
    if [ ! -d "og_export" ]; then
        echo "Downloading OmniGlue..."
        if [ -f "og_export.zip" ]; then
            # Extract if already downloaded
            unzip -q og_export.zip
            rm og_export.zip
        else
            # Download and extract
            wget -q https://storage.googleapis.com/omniglue/og_export.zip
            unzip -q og_export.zip
            rm og_export.zip
        fi
        echo "OmniGlue model downloaded and extracted."
    else
        echo "OmniGlue model already exists."
    fi
    
    cd ../../../../
}

# Call the function to download models
download_models

# Create configuration file for the Python backend with additional parameters
echo "Creating configuration file..."
cat > model_repository/omniglue_python/config.pbtxt << EOL
name: "omniglue_python"
backend: "python"
max_batch_size: 0
input [
  {
    name: "image0"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
  },
  {
    name: "image1"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
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

# Optional parameters
parameters {
  key: "max_keypoints"
  value: {
    string_value: "1024"
  }
}

# Instance groups - adjust based on your hardware
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
EOL

# Create a requirements.txt file
echo "Creating requirements.txt file..."
cat > model_repository/omniglue_python/1/requirements.txt << EOL
numpy>=1.19.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
pillow>=8.0.0
EOL

# Create a Dockerfile for the custom Triton image
echo "Creating Dockerfile for custom Triton image..."
cat > Dockerfile << EOL
FROM nvcr.io/nvidia/tritonserver:23.04-py3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt

# Install Python dependencies
COPY model_repository/omniglue_python/1/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

# Clone and install OmniGlue
RUN git clone https://github.com/google-research/omniglue.git /opt/omniglue
WORKDIR /opt/omniglue
RUN pip install -e .

# Set environment variable for the OmniGlue model path
ENV OMNIGLUE_MODEL_PATH=/models/omniglue_python/1/models
ENV PYTHONPATH=/opt/omniglue:\${PYTHONPATH}

# Back to root directory
WORKDIR /
EOL

# Create script to build and start the custom Triton container
echo "Creating script to build and start custom Triton..."
cat > build_and_start_triton.sh << EOL
#!/bin/bash
# Build and start the custom Triton container with OmniGlue Python backend

# Copy the improved Python backend to the model directory
cp improved_omniglue_backend.py model_repository/omniglue_python/1/model.py

# Build the custom image
echo "Building custom Triton image with OmniGlue..."
docker build -t triton-omniglue .

# Run the container with verbose logging
echo "Starting Triton server with OmniGlue backend..."
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v \${PWD}/model_repository:/models \\
  --name triton-omniglue-server \\
  triton-omniglue \\
  tritonserver --model-repository=/models --log-verbose=1 --log-info=1
EOL

chmod +x build_and_start_triton.sh

# Create a client script for testing
echo "Creating test client script..."
cat > test_omniglue_client.py << EOL
#!/usr/bin/env python3
"""
Client script to test OmniGlue Python backend in Triton
"""

import os
import sys
import numpy as np
import cv2
import argparse
import time
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=None):
    """
    Preprocess an image for OmniGlue.
    
    Args:
        image_path: Path to the image file
        target_size: Optional tuple (width, height) to resize the image
        
    Returns:
        Preprocessed image as numpy array in CHW format
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Resize if target_size is specified
    if target_size:
        img = cv2.resize(img, target_size)
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Transpose to CHW format (channels, height, width)
    img = img.transpose(2, 0, 1)
    
    return img

def visualize_matches(image1_path, image2_path, matches_kp0, matches_kp1, 
                     match_confidences, output_path=None, max_matches=50):
    """
    Visualize the matches between two images.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        matches_kp0: Keypoints from the first image
        matches_kp1: Keypoints from the second image
        match_confidences: Confidence scores for each match
        output_path: Path to save the visualization (optional)
        max_matches: Maximum number of matches to visualize
    """
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Resize if needed to match the size used in preprocessing
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Convert BGR to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create a figure
    plt.figure(figsize=(20, 10))
    
    # Display concatenated images
    concat_img = np.hstack((img1_rgb, img2_rgb))
    plt.imshow(concat_img)
    plt.axis('off')
    
    # Draw matches
    num_matches = min(len(match_confidences), max_matches)
    
    # Sort matches by confidence
    if match_confidences is not None and len(match_confidences) > 0:
        indices = np.argsort(match_confidences)[::-1]  # Sort in descending order
        top_indices = indices[:num_matches]
        
        # Draw lines between matches
        for i in top_indices:
            x1, y1 = matches_kp0[i]
            x2, y2 = matches_kp1[i]
            x2 = x2 + w1  # Adjust x-coordinate for the second image
            
            # Draw a line with color based on confidence
            conf = match_confidences[i]
            # Higher confidence = more green, lower confidence = more red
            color = (0, min(conf * 2, 1.0), max(0, 1 - conf * 2))
            
            plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1)
            plt.plot(x1, y1, 'o', color=color, markersize=5)
            plt.plot(x2, y2, 'o', color=color, markersize=5)
    
        plt.title(f"OmniGlue Matching Results: Top {num_matches} matches")
        
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    else:
        print("No matches to visualize.")

def main():
    parser = argparse.ArgumentParser(description="Test OmniGlue Python backend in Triton")
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--server-url', type=str, default='localhost:8000', help='Triton server URL')
    parser.add_argument('--output', type=str, default='matches.png', help='Output path for visualization')
    parser.add_argument('--width', type=int, default=640, help='Image width for preprocessing')
    parser.add_argument('--height', type=int, default=480, help='Image height for preprocessing')
    args = parser.parse_args()
    
    try:
        # Check if images exist
        if not os.path.exists(args.image1):
            raise FileNotFoundError(f"Image not found: {args.image1}")
        if not os.path.exists(args.image2):
            raise FileNotFoundError(f"Image not found: {args.image2}")
        
        # Preprocess images
        print("Preprocessing images...")
        target_size = (args.width, args.height)
        image0 = preprocess_image(args.image1, target_size)
        image1 = preprocess_image(args.image2, target_size)
        
        # Initialize the client
        print(f"Connecting to Triton server at {args.server_url}...")
        client = InferenceServerClient(url=args.server_url)
        
        # Check server status
        if not client.is_server_live():
            raise ConnectionError("Triton server is not live")
        
        # Check if our model is ready
        model_name = "omniglue_python"
        if not client.is_model_ready(model_name):
            raise RuntimeError(f"{model_name} model is not ready")
        
        # Get model metadata for debugging
        model_metadata = client.get_model_metadata(model_name)
        print(f"Model inputs: {[input.name for input in model_metadata.inputs]}")
        print(f"Model outputs: {[output.name for output in model_metadata.outputs]}")
        
        print("Server is ready. Creating inference request...")
        
        # Create input tensors
        input0 = InferInput("image0", image0.shape, "FP32")
        input0.set_data_from_numpy(image0)
        
        input1 = InferInput("image1", image1.shape, "FP32")
        input1.set_data_from_numpy(image1)
        
        # Specify output names
        outputs = [
            InferRequestedOutput("matches_kp0"),
            InferRequestedOutput("matches_kp1"),
            InferRequestedOutput("match_confidences")
        ]
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        response = client.infer(model_name, [input0, input1], outputs=outputs)
        inference_time = time.time() - start_time
        
        # Get results
        matches_kp0 = response.as_numpy("matches_kp0")
        matches_kp1 = response.as_numpy("matches_kp1")
        match_confidences = response.as_numpy("match_confidences")
        
        print(f"Inference completed in {inference_time:.2f} seconds")
        print(f"Found {len(match_confidences)} matches")
        
        # Show top 5 matches
        if len(match_confidences) > 0:
            top_indices = np.argsort(match_confidences)[-5:][::-1]
            print("\nTop 5 matches:")
            for i, idx in enumerate(top_indices):
                print(f"  Match {i+1}: confidence={match_confidences[idx]:.4f}")
                print(f"    Point in image 1: ({matches_kp0[idx][0]:.1f}, {matches_kp0[idx][1]:.1f})")
                print(f"    Point in image 2: ({matches_kp1[idx][0]:.1f}, {matches_kp1[idx][1]:.1f})")
        
        # Visualize matches
        print("\nVisualizing matches...")
        visualize_matches(args.image1, args.image2, matches_kp0, matches_kp1, 
                         match_confidences, args.output)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOL

# Create helpful README
echo "Creating README..."
cat > README.md << EOL
# OmniGlue in Triton Inference Server (Python Backend)

This repository provides a Python backend implementation of OmniGlue for NVIDIA Triton Inference Server.

## Overview

OmniGlue is a feature matching model that uses foundation model guidance (DINOv2) to improve generalization across different image domains. This implementation uses Triton's Python backend to run the original OmniGlue implementation directly, avoiding ONNX conversion issues.

## Setup Instructions

1. Copy the improved Python backend to this directory:
   \`\`\`bash
   cp /path/to/improved_omniglue_backend.py .
   \`\`\`

2. Build and start the Triton server:
   \`\`\`bash
   ./build_and_start_triton.sh
   \`\`\`

3. Test with your own images:
   \`\`\`bash
   python test_omniglue_client.py --image1 /path/to/image1.jpg --image2 /path/to/image2.jpg
   \`\`\`

## Model Architecture

The implementation follows the OmniGlue architecture which consists of:

1. **SuperPoint**: Extracts keypoints and descriptors from input images
2. **DINOv2**: A foundation model that provides generalizable visual guidance
3. **OmniGlue**: The matching model that combines the above components

## Configuration

The backend supports the following configuration parameters:

- \`max_keypoints\`: Maximum number of keypoints to use (default: 1024)

You can modify these parameters in the \`config.pbtxt\` file.

## Troubleshooting

- Check Triton server logs for detailed error messages
- Verify that model files are correctly downloaded and extracted
- Ensure your input images can be properly read and processed
EOL

echo "Setup complete! Next steps:"
echo "1. Copy the improved_omniglue_backend.py file to this directory"
echo "2. Build and start Triton server: ./build_and_start_triton.sh"
echo "3. Test with your images: python test_omniglue_client.py --image1 /path/to/image1.jpg --image2 /path/to/image2.jpg"
