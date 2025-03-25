# OmniGlue in NVIDIA Triton Inference Server

This repository contains scripts to deploy [OmniGlue](https://github.com/google-research/omniglue) (a generalizable feature matching model) in NVIDIA Triton Inference Server as an ensemble model that combines SuperPoint, DINOv2, and OmniGlue.

## Overview

OmniGlue is a feature matching model that leverages vision foundation models (specifically DINOv2) to guide the matching process. The complete pipeline consists of:

1. **SuperPoint**: Extracts keypoints and descriptors from input images
2. **DINOv2**: Provides foundation model features to guide the matching
3. **OmniGlue**: Uses the keypoints, descriptors, and foundation model features to perform matching

This repository provides scripts to set up this pipeline as an ensemble in NVIDIA Triton Inference Server, allowing efficient inference with a single client request.

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit installed
- Python 3.8 or higher
- PyTorch 1.9 or higher
- ONNX Runtime
- Triton Client libraries

## Repository Structure

```
.
├── deploy_omniglue_triton.sh         # Main deployment script
├── convert_models_to_onnx.py         # Script to convert models to ONNX format
├── test_client.py                    # Client script to test the deployment
├── README.md                         # This file
├── model_repository/                 # Created by deploy script
│   ├── superpoint/                   # SuperPoint model
│   ├── dinov2/                       # DINOv2 model
│   ├── omniglue/                     # OmniGlue model
│   └── omniglue_ensemble/            # Ensemble configuration
└── start_triton.sh                   # Script to start Triton server
```

## Installation and Deployment

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/omniglue-triton.git
   cd omniglue-triton
   ```

2. Run the deployment script:
   ```bash
   bash deploy_omniglue_triton.sh
   ```
   This script will:
   - Create the necessary directory structure
   - Download SuperPoint, DINOv2, and OmniGlue models
   - Convert the models to ONNX format
   - Create Triton configuration files for each model and the ensemble
   - Create a script to start the Triton server

3. Start the Triton server:
   ```bash
   ./start_triton.sh
   ```

4. Test the deployment:
   ```bash
   python test_client.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
   ```

## Model Ensemble Architecture

The OmniGlue ensemble in Triton is structured as follows:

```
Input: image0, image1
│
├─> SuperPoint(image0) ──> keypoints0, descriptors0
│
├─> SuperPoint(image1) ──> keypoints1, descriptors1
│
├─> DINOv2(image0) ────> features0
│
├─> DINOv2(image1) ────> features1
│
└─> OmniGlue(keypoints0, descriptors0, keypoints1, descriptors1, features0, features1)
    │
    └─> matches_kp0, matches_kp1, match_confidences
```

## Performance Optimization

For better performance, you can adjust the following parameters in the model configuration files:

- `instance_group`: Control the number of model instances per GPU
- `dynamic_batching`: Configure batch settings for higher throughput
- Backend-specific parameters: ONNX Runtime provides additional optimization options

## Troubleshooting

If you encounter issues:

1. Check Triton server logs: `docker logs <container_id>`
2. Verify model conversion was successful
3. Test each model individually before using the ensemble
4. Check tensor shapes in the configuration files match the models' requirements

## References

- [OmniGlue Paper](https://arxiv.org/abs/2312.00307)
- [OmniGlue GitHub Repository](https://github.com/google-research/omniglue)
- [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [ONNX-Compatible OmniGlue Repository](https://github.com/datomi79/omniglue-onnx)

## License

This repository is for educational purposes. Please respect the licenses of OmniGlue, SuperPoint, DINOv2, and NVIDIA Triton Inference Server.
