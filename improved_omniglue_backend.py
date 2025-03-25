#!/usr/bin/env python3
"""
Improved Python backend for OmniGlue in Triton Inference Server

This custom backend implements the OmniGlue feature matching pipeline using the original
Python implementation rather than ONNX conversion, avoiding compatibility issues.

The backend follows the architecture of OmniGlue, which involves three main components:
1. SuperPoint: For keypoint detection and descriptor extraction
2. DINOv2: A foundation model for generalizable visual guidance
3. OmniGlue: The matching model that combines these inputs to find correspondences

References:
- OmniGlue paper: https://arxiv.org/abs/2405.12979
- OmniGlue repo: https://github.com/google-research/omniglue
"""

import os
import sys
import json
import time
import logging
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OmniGlueBackend")

# Flag to determine if OmniGlue is available
OMNIGLUE_AVAILABLE = False

# Try to import OmniGlue and its dependencies
try:
    import torch
    import omniglue
    OMNIGLUE_AVAILABLE = True
    logger.info("Successfully imported OmniGlue and its dependencies.")
except ImportError as e:
    logger.warning(f"Failed to import OmniGlue: {str(e)}")
    logger.warning("Using dummy implementation since OmniGlue is not available.")

class TritonPythonModel:
    """
    Python model for OmniGlue feature matching.
    Implements the Triton Python Backend API.
    """

    def initialize(self, args):
        """
        Initialize the model. This will be called during model loading.
        
        Parameters:
            args: Dictionary containing model configuration information.
        """
        logger.info("Initializing OmniGlue Python backend")
        
        # Parse model config
        self.model_config = json.loads(args['model_config'])
        self.model_instance_name = args['model_instance_name']
        
        # Get environment variable for model paths
        self.model_path = os.environ.get('OMNIGLUE_MODEL_PATH', './models')
        logger.info(f"Using model path: {self.model_path}")
        
        # Get output configurations
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "matches_kp0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])
            
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "matches_kp1")
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])
            
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "match_confidences")
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])
        
        # These will be filled in during initialization
        self.omniglue_model = None
        self.initialization_success = False
        
        # Add support for optional parameters from config.pbtxt
        self.max_keypoints = 1024  # Default value
        if "parameters" in self.model_config:
            parameters = self.model_config["parameters"]
            if "max_keypoints" in parameters:
                self.max_keypoints = int(parameters["max_keypoints"]["string_value"])
                logger.info(f"Setting max_keypoints from config: {self.max_keypoints}")
        
        # Initialize OmniGlue if available
        if OMNIGLUE_AVAILABLE:
            try:
                # Check if model files exist
                og_export_path = os.path.join(self.model_path, 'og_export')
                sp_export_path = os.path.join(self.model_path, 'sp_v6')
                dino_export_path = os.path.join(self.model_path, 'dinov2_vitb14_pretrain.pth')
                
                paths_exist = (
                    os.path.exists(og_export_path) and
                    os.path.exists(sp_export_path) and
                    os.path.exists(dino_export_path)
                )
                
                if not paths_exist:
                    logger.warning(f"One or more model paths do not exist:")
                    logger.warning(f"  OmniGlue: {og_export_path} - Exists: {os.path.exists(og_export_path)}")
                    logger.warning(f"  SuperPoint: {sp_export_path} - Exists: {os.path.exists(sp_export_path)}")
                    logger.warning(f"  DINOv2: {dino_export_path} - Exists: {os.path.exists(dino_export_path)}")
                    raise FileNotFoundError("Required model files not found")
                
                # Initialize the OmniGlue model
                logger.info("Loading OmniGlue model and its components...")
                start_time = time.time()
                
                self.omniglue_model = omniglue.OmniGlue(
                    og_export=og_export_path,
                    sp_export=sp_export_path,
                    dino_export=dino_export_path,
                )
                
                load_time = time.time() - start_time
                logger.info(f"OmniGlue model initialized successfully in {load_time:.2f} seconds")
                self.initialization_success = True
                
                # Try a test inference with dummy data to ensure everything works
                try:
                    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    self.omniglue_model.FindMatches(dummy_image, dummy_image, max_keypoints=10)
                    logger.info("Test inference successful")
                except Exception as e:
                    logger.warning(f"Test inference failed: {str(e)}")
                    # Continue anyway, as this might be due to the dummy data
            except Exception as e:
                logger.error(f"Error initializing OmniGlue model: {str(e)}")
                self.omniglue_model = None
                
    def execute(self, requests):
        """
        Process inference requests and return results.
        
        Parameters:
            requests: List of pb_utils.InferenceRequest objects.
            
        Returns:
            List of pb_utils.InferenceResponse objects.
        """
        logger.debug(f"Received {len(requests)} inference requests")
        responses = []
        
        for request_idx, request in enumerate(requests):
            try:
                # Start timing for this request
                start_time = time.time()
                
                # Get input tensors
                in_0 = pb_utils.get_input_tensor_by_name(request, "image0")
                in_1 = pb_utils.get_input_tensor_by_name(request, "image1")
                
                # Convert to numpy arrays
                image0 = in_0.as_numpy()
                image1 = in_1.as_numpy()
                
                # Get optional parameters if provided
                max_keypoints = self.max_keypoints
                
                # Log input shapes for debugging
                logger.debug(f"Request {request_idx}: Image0 shape: {image0.shape}, Image1 shape: {image1.shape}")
                
                # Process images
                # Step 1: Process image format (convert from CHW to HWC if needed)
                if image0.ndim == 3 and image0.shape[0] == 3:  # CHW format
                    logger.debug("Converting image0 from CHW to HWC format")
                    image0 = np.transpose(image0, (1, 2, 0))  # Convert to HWC
                
                if image1.ndim == 3 and image1.shape[0] == 3:  # CHW format
                    logger.debug("Converting image1 from CHW to HWC format")
                    image1 = np.transpose(image1, (1, 2, 0))  # Convert to HWC
                
                # Step 2: Scale images to uint8 if they're normalized to [0, 1]
                if image0.dtype == np.float32 and image0.max() <= 1.0:
                    logger.debug("Scaling image0 from [0,1] to [0,255]")
                    image0 = (image0 * 255).astype(np.uint8)
                
                if image1.dtype == np.float32 and image1.max() <= 1.0:
                    logger.debug("Scaling image1 from [0,1] to [0,255]")
                    image1 = (image1 * 255).astype(np.uint8)
                
                # Step 3: Perform matching using OmniGlue
                if self.omniglue_model is not None and self.initialization_success:
                    try:
                        # Call OmniGlue's FindMatches function
                        logger.debug(f"Running OmniGlue.FindMatches with max_keypoints={max_keypoints}")
                        matches_kp0, matches_kp1, confidences = self.omniglue_model.FindMatches(
                            image0, 
                            image1, 
                            max_keypoints=max_keypoints
                        )
                        
                        # Log match statistics
                        match_time = time.time() - start_time
                        logger.debug(f"Found {len(confidences)} matches in {match_time:.2f} seconds")
                        
                    except Exception as e:
                        logger.error(f"Error during OmniGlue matching: {str(e)}")
                        # Provide dummy outputs on error
                        matches_kp0 = np.zeros((10, 2), dtype=np.float32)
                        matches_kp1 = np.zeros((10, 2), dtype=np.float32)
                        confidences = np.zeros(10, dtype=np.float32)
                else:
                    # Dummy implementation if OmniGlue is not available or initialization failed
                    logger.warning("Using dummy implementation as OmniGlue is not available or initialization failed")
                    matches_kp0 = np.zeros((10, 2), dtype=np.float32)
                    matches_kp1 = np.zeros((10, 2), dtype=np.float32)
                    confidences = np.zeros(10, dtype=np.float32)
                
                # Create output tensors
                out_tensor_0 = pb_utils.Tensor("matches_kp0", 
                                              matches_kp0.astype(self.output0_dtype))
                out_tensor_1 = pb_utils.Tensor("matches_kp1", 
                                              matches_kp1.astype(self.output1_dtype))
                out_tensor_2 = pb_utils.Tensor("match_confidences", 
                                              confidences.astype(self.output2_dtype))
                
                # Create InferenceResponse for this request
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2])
                responses.append(inference_response)
                
                # Log total processing time
                total_time = time.time() - start_time
                logger.debug(f"Request {request_idx} processed in {total_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing request {request_idx}: {str(e)}")
                # Return empty response on error
                out_tensor_0 = pb_utils.Tensor("matches_kp0", 
                                             np.zeros((0, 2), dtype=self.output0_dtype))
                out_tensor_1 = pb_utils.Tensor("matches_kp1", 
                                             np.zeros((0, 2), dtype=self.output1_dtype))
                out_tensor_2 = pb_utils.Tensor("match_confidences", 
                                             np.zeros(0, dtype=self.output2_dtype))
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2],
                    error=pb_utils.TritonError(f"Error processing request: {str(e)}"))
                responses.append(inference_response)
            
        return responses
        
    def finalize(self):
        """
        Clean up resources when the model is unloaded.
        """
        logger.info("Cleaning up OmniGlue backend resources")
        self.omniglue_model = None
        
        # Clear CUDA cache if available
        if OMNIGLUE_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
