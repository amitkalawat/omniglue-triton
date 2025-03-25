#!/usr/bin/env python3
"""
Client script to test OmniGlue ensemble in Triton Inference Server
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import argparse

def preprocess_image(image_path, target_size=(640, 480)):
    """Preprocess an image for the OmniGlue ensemble."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
        
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Convert to RGB (from BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Transpose to CHW format (channels, height, width)
    img = img.transpose(2, 0, 1)
    
    return img

def visualize_matches(image1_path, image2_path, matches_kp0, matches_kp1, confidences, output_path=None):
    """Visualize the matches between two images."""
    # Read the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Resize if needed
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    
    # Create a figure
    plt.figure(figsize=(20, 10))
    
    # Convert BGR to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Concatenate images horizontally
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Display both images
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title("Image 1")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title("Image 2")
    plt.axis('off')
    
    # Draw matches
    # Getting top N matches for visualization
    num_matches = min(len(confidences), 50)  # Limit to top 50 matches
    
    # Sort matches by confidence
    if confidences is not None and len(confidences) > 0:
        indices = np.argsort(confidences)[::-1]  # Sort in descending order
        top_indices = indices[:num_matches]
        
        # Draw lines between matches
        plt.figure(figsize=(20, 10))
        plt.imshow(np.hstack((img1_rgb, img2_rgb)))
        plt.axis('off')
        
        for i in top_indices:
            x1, y1 = matches_kp0[i]
            x2, y2 = matches_kp1[i]
            x2 = x2 + w1  # Adjust x-coordinate for the second image
            
            # Draw a line with color based on confidence (red: low, green: high)
            conf = confidences[i]
            color = (0, conf, 1-conf)  # RGB: more green for higher confidence
            
            plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1)
            plt.plot(x1, y1, 'o', color=color, markersize=3)
            plt.plot(x2, y2, 'o', color=color, markersize=3)
        
        plt.title(f"Top {num_matches} matches by confidence")
        
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    else:
        print("No matches to visualize.")

def main():
    parser = argparse.ArgumentParser(description="Test OmniGlue in Triton Inference Server")
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--server-url', type=str, default='localhost:8000', help='Triton server URL')
    parser.add_argument('--output', type=str, default='matches.png', help='Output visualization path')
    args = parser.parse_args()
    
    try:
        # Check if images exist
        if not os.path.exists(args.image1):
            raise FileNotFoundError(f"Image not found: {args.image1}")
        if not os.path.exists(args.image2):
            raise FileNotFoundError(f"Image not found: {args.image2}")
            
        # Preprocess images
        print("Preprocessing images...")
        image0 = preprocess_image(args.image1)
        image1 = preprocess_image(args.image2)
        
        # Initialize the client
        print(f"Connecting to Triton server at {args.server_url}...")
        client = InferenceServerClient(url=args.server_url)
        
        # Check server status
        if not client.is_server_live():
            raise ConnectionError("Triton server is not live")
        
        # Check if our model is ready
        if not client.is_model_ready("omniglue_ensemble"):
            raise RuntimeError("OmniGlue ensemble model is not ready")
            
        print("Server is ready. Creating inference request...")
        
        # Create input tensors for the ensemble model
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
        
        # Run inference through the ensemble
        print("Running inference...")
        response = client.infer("omniglue_ensemble", [input0, input1], outputs=outputs)
        
        # Get results
        matches_kp0 = response.as_numpy("matches_kp0")
        matches_kp1 = response.as_numpy("matches_kp1")
        match_confidences = response.as_numpy("match_confidences")
        
        print(f"Found {len(match_confidences)} matches")
        
        # Visualize matches
        print("Visualizing matches...")
        visualize_matches(args.image1, args.image2, matches_kp0, matches_kp1, match_confidences, args.output)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
