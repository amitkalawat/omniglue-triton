#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU processing for OmniGlue feature matching.
This script provides functions for leveraging multiple GPUs to process large images
or multiple image pairs with OmniGlue.
"""

import os
import gc
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import sys
import tempfile

def match_images_multi_gpu(image_pairs, gpu_ids=None, max_size=1024):
    """Process multiple image pairs in parallel across multiple GPUs.
    
    Args:
        image_pairs: List of tuples [(image0_path, image1_path), ...]
        gpu_ids: List of GPU IDs to use [0, 1, 2, 3]. If None, use all available GPUs.
        max_size: Maximum dimension for images
        
    Returns:
        results: List of (matches, visualization) tuples for each image pair
    """
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        # Validate GPU IDs
        for gpu_id in gpu_ids:
            if gpu_id >= num_gpus:
                raise ValueError(f"GPU ID {gpu_id} is not available. Only {num_gpus} GPUs found.")
    
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    
    # Function to process a single image pair on a specific GPU
    def process_pair(args):
        pair_idx, gpu_id, image0_path, image1_path = args
        
        # Set the GPU to use
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # Always use device 0 since we've set CUDA_VISIBLE_DEVICES
        
        print(f"Processing pair {pair_idx+1}/{len(image_pairs)} on GPU {gpu_id}")
        
        # Use our optimized function but with larger max_size since we have dedicated GPU
        from match_images_low_memory import match_images_low_memory
        matches, visualization = match_images_low_memory(
            image0_path, image1_path, 
            visualize=True, 
            max_size=max_size
        )
        
        return matches, visualization
    
    # Prepare arguments for each process
    args_list = []
    for i, (image0_path, image1_path) in enumerate(image_pairs):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Distribute work in round-robin fashion
        args_list.append((i, gpu_id, image0_path, image1_path))
    
    # Process image pairs in parallel
    if len(image_pairs) > 1:
        # Use multiprocessing for multiple pairs
        mp.set_start_method('spawn', force=True)
        with mp.Pool(len(gpu_ids)) as pool:
            results = pool.map(process_pair, args_list)
    else:
        # Just process directly for a single pair
        results = [process_pair(args_list[0])]
    
    return results


def match_large_images_multi_gpu(image0_path, image1_path, gpu_ids=None, tile_size=2048, overlap=256):
    """Process large images by splitting them into tiles and processing across multiple GPUs.
    
    Args:
        image0_path: Path to the first image
        image1_path: Path to the second image
        gpu_ids: List of GPU IDs to use [0, 1, 2, 3]. If None, use all available GPUs.
        tile_size: Size of each tile to process
        overlap: Overlap between tiles to ensure features near boundaries are matched
        
    Returns:
        matches: Combined matches from all tiles
        visualization: Visualization of the matches
    """
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        # Validate GPU IDs
        for gpu_id in gpu_ids:
            if gpu_id >= num_gpus:
                raise ValueError(f"GPU ID {gpu_id} is not available. Only {num_gpus} GPUs found.")
    
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    
    # Load the full images
    image0 = np.array(Image.open(image0_path).convert('RGB'))
    image1 = np.array(Image.open(image1_path).convert('RGB'))
    
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    
    print(f"Image 0 dimensions: {w0}x{h0}")
    print(f"Image 1 dimensions: {w1}x{h1}")
    
    # Create tiles with overlap
    def create_tiles(image, tile_size, overlap):
        h, w = image.shape[:2]
        tiles = []
        positions = []
        
        y_steps = max(1, h // (tile_size - overlap))
        x_steps = max(1, w // (tile_size - overlap))
        
        for y in range(y_steps):
            y_start = min(y * (tile_size - overlap), max(0, h - tile_size))
            y_end = min(y_start + tile_size, h)
            
            for x in range(x_steps):
                x_start = min(x * (tile_size - overlap), max(0, w - tile_size))
                x_end = min(x_start + tile_size, w)
                
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append(tile)
                positions.append((x_start, y_start, x_end, y_end))
        
        return tiles, positions
    
    # Create tiles for both images
    tiles0, positions0 = create_tiles(image0, tile_size, overlap)
    tiles1, positions1 = create_tiles(image1, tile_size, overlap)
    
    print(f"Created {len(tiles0)} tiles for image 0")
    print(f"Created {len(tiles1)} tiles for image 1")
    
    # Function to process a pair of tiles on a specific GPU
    def process_tile_pair(args):
        pair_idx, gpu_id, tile0, pos0, tile1, pos1 = args
        
        # Set the GPU to use
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # Always use device 0 since we've set CUDA_VISIBLE_DEVICES
        
        print(f"Processing tile pair {pair_idx+1}/{len(tiles0)*len(tiles1)} on GPU {gpu_id}")
        
        # Save tiles temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f0, \
             tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
            tile0_path = f0.name
            tile1_path = f1.name
            
            Image.fromarray(tile0).save(tile0_path)
            Image.fromarray(tile1).save(tile1_path)
        
        try:
            # Use our optimized function
            from match_images_low_memory import match_images_low_memory
            matches, _ = match_images_low_memory(
                tile0_path, tile1_path, 
                visualize=False,  # Don't visualize individual tiles
                max_size=tile_size  # No need to resize
            )
            
            # Adjust keypoint coordinates to the original image coordinates
            if matches[0] is not None:
                x0_offset, y0_offset, _, _ = pos0
                x1_offset, y1_offset, _, _ = pos1
                
                match_kp0, match_kp1, match_confidences = matches
                
                # Adjust coordinates
                match_kp0[:, 0] += x0_offset
                match_kp0[:, 1] += y0_offset
                match_kp1[:, 0] += x1_offset
                match_kp1[:, 1] += y1_offset
                
                return match_kp0, match_kp1, match_confidences
            else:
                return None, None, None
            
        finally:
            # Clean up temporary files
            os.unlink(tile0_path)
            os.unlink(tile1_path)
    
    # Prepare arguments for each process
    args_list = []
    pair_idx = 0
    for i, (tile0, pos0) in enumerate(zip(tiles0, positions0)):
        for j, (tile1, pos1) in enumerate(zip(tiles1, positions1)):
            gpu_id = gpu_ids[pair_idx % len(gpu_ids)]  # Distribute work in round-robin fashion
            args_list.append((pair_idx, gpu_id, tile0, pos0, tile1, pos1))
            pair_idx += 1
    
    # Process tile pairs in parallel
    mp.set_start_method('spawn', force=True)
    with mp.Pool(len(gpu_ids)) as pool:
        results = pool.map(process_tile_pair, args_list)
    
    # Combine results from all tiles
    all_kp0 = []
    all_kp1 = []
    all_confidences = []
    
    for match_kp0, match_kp1, match_confidences in results:
        if match_kp0 is not None:
            all_kp0.append(match_kp0)
            all_kp1.append(match_kp1)
            all_confidences.append(match_confidences)
    
    if all_kp0:
        combined_kp0 = np.vstack(all_kp0)
        combined_kp1 = np.vstack(all_kp1)
        combined_confidences = np.concatenate(all_confidences)
        
        # Remove duplicate matches (keypoints that are very close to each other)
        # This is needed because of the overlap between tiles
        try:
            from sklearn.cluster import DBSCAN
            
            # Cluster keypoints in image 0
            clustering0 = DBSCAN(eps=10, min_samples=1).fit(combined_kp0)
            labels0 = clustering0.labels_
            
            # For each cluster, keep only the match with highest confidence
            unique_labels0 = np.unique(labels0)
            filtered_kp0 = []
            filtered_kp1 = []
            filtered_confidences = []
            
            for label in unique_labels0:
                indices = np.where(labels0 == label)[0]
                best_idx = indices[np.argmax(combined_confidences[indices])]
                filtered_kp0.append(combined_kp0[best_idx])
                filtered_kp1.append(combined_kp1[best_idx])
                filtered_confidences.append(combined_confidences[best_idx])
            
            filtered_kp0 = np.array(filtered_kp0)
            filtered_kp1 = np.array(filtered_kp1)
            filtered_confidences = np.array(filtered_confidences)
        except ImportError:
            # If sklearn is not available, use a simpler approach
            print("Warning: sklearn not available, using all matches without deduplication")
            filtered_kp0 = combined_kp0
            filtered_kp1 = combined_kp1
            filtered_confidences = combined_confidences
        
        print(f"Found {len(filtered_kp0)} unique matches across all tiles")
        
        # Create visualization
        from omniglue import utils
        visualization = utils.visualize_matches(
            image0, image1, filtered_kp0, filtered_kp1, 
            np.eye(len(filtered_kp0)),
            show_keypoints=True,
            highlight_unmatched=True,
            title=f"{len(filtered_kp0)} matches",
            line_width=1,
        )
        
        plt.figure(figsize=(15, 10))
        plt.imshow(visualization)
        plt.axis('off')
        plt.title(f"OmniGlue: {os.path.basename(image0_path)} ↔ {os.path.basename(image1_path)}")
        plt.tight_layout()
        plt.show()
        
        return (filtered_kp0, filtered_kp1, filtered_confidences), visualization
    else:
        print("No matches found in any tile")
        return None, None


def match_images_model_parallel(image0_path, image1_path, visualize=True, max_size=2048):
    """Process images using model parallelism across multiple GPUs.
    
    This function distributes different parts of the OmniGlue pipeline across different GPUs:
    - GPU 0: SuperPoint for both images
    - GPU 1: DINOv2 for both images
    - GPU 2: OmniGlue matching
    - GPU 3: Visualization (if available)
    
    Args:
        image0_path: Path to the first image
        image1_path: Path to the second image
        visualize: Whether to visualize the matches
        max_size: Maximum dimension for images
        
    Returns:
        matches: Matched keypoints
        visualization: Visualization of the matches if visualize=True
    """
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")
    
    print(f"Found {num_gpus} GPUs")
    
    # Import OmniGlue correctly
    sys.path.append('/Users/kwt/Documents/2025/q-cli-client-testing/omniglue-triton')
    from omniglue.src.omniglue import OmniGlue
    from omniglue import utils
    
    # Set environment variable for PyTorch memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Load and resize images if needed
    def load_and_resize_image(path):
        img = Image.open(path).convert('RGB')
        # Resize if too large (preserving aspect ratio)
        if max(img.size) > max_size:
            scale = max_size / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        return np.array(img)
    
    # Load images
    print("> Loading and resizing images...")
    image0 = load_and_resize_image(image0_path)
    image1 = load_and_resize_image(image1_path)
    
    print(f"> Image dimensions: {image0.shape} and {image1.shape}")
    
    try:
        # Create custom extractors for each GPU
        from omniglue.src.omniglue.superpoint_extract import SuperPointExtract
        from omniglue.src.omniglue.dino_extract import DINOExtract
        
        # GPU 0: SuperPoint
        print("> Setting up SuperPoint on GPU 0...")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        sp_extract = SuperPointExtract("./omniglue/models/sp_v6")
        
        # GPU 1: DINOv2 (if available, otherwise use GPU 0)
        dino_gpu = min(1, num_gpus - 1)
        print(f"> Setting up DINOv2 on GPU {dino_gpu}...")
        torch.cuda.set_device(dino_gpu)
        torch.cuda.empty_cache()
        dino_extract = DINOExtract("./omniglue/models/dinov2_vitb14_pretrain.pth", feature_layer=1)
        
        # GPU 2: OmniGlue (if available, otherwise use GPU 0)
        og_gpu = min(2, num_gpus - 1)
        print(f"> Setting up OmniGlue on GPU {og_gpu}...")
        torch.cuda.set_device(og_gpu)
        torch.cuda.empty_cache()
        
        import tensorflow as tf
        og_matcher = tf.saved_model.load("./omniglue/models/og_export")
        
        # Extract features on respective GPUs
        print("> Extracting SuperPoint features (GPU 0)...")
        torch.cuda.set_device(0)
        start_time = time.time()
        sp_features0 = sp_extract(image0)
        sp_features1 = sp_extract(image1)
        print(f"> \tSuperPoint extraction took {time.time() - start_time:.2f} seconds")
        
        print(f"> Extracting DINOv2 features (GPU {dino_gpu})...")
        torch.cuda.set_device(dino_gpu)
        start_time = time.time()
        dino_features0 = dino_extract(image0)
        dino_features1 = dino_extract(image1)
        print(f"> \tDINOv2 extraction took {time.time() - start_time:.2f} seconds")
        
        # Process DINOv2 descriptors
        height0, width0 = image0.shape[:2]
        height1, width1 = image1.shape[:2]
        
        dino_descriptors0 = dino_extract.get_dino_descriptors(
            dino_features0,
            tf.convert_to_tensor(sp_features0[0], dtype=tf.float32),
            tf.convert_to_tensor(height0, dtype=tf.int32),
            tf.convert_to_tensor(width0, dtype=tf.int32),
            768,  # DINO_FEATURE_DIM
        )
        dino_descriptors1 = dino_extract.get_dino_descriptors(
            dino_features1,
            tf.convert_to_tensor(sp_features1[0], dtype=tf.float32),
            tf.convert_to_tensor(height1, dtype=tf.int32),
            tf.convert_to_tensor(width1, dtype=tf.int32),
            768,  # DINO_FEATURE_DIM
        )
        
        # Construct inputs for OmniGlue
        inputs = {
            'keypoints0': tf.convert_to_tensor(
                np.expand_dims(sp_features0[0], axis=0),
                dtype=tf.float32,
            ),
            'keypoints1': tf.convert_to_tensor(
                np.expand_dims(sp_features1[0], axis=0), dtype=tf.float32
            ),
            'descriptors0': tf.convert_to_tensor(
                np.expand_dims(sp_features0[1], axis=0), dtype=tf.float32
            ),
            'descriptors1': tf.convert_to_tensor(
                np.expand_dims(sp_features1[1], axis=0), dtype=tf.float32
            ),
            'scores0': tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(sp_features0[2], axis=0), axis=-1),
                dtype=tf.float32,
            ),
            'scores1': tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(sp_features1[2], axis=0), axis=-1),
                dtype=tf.float32,
            ),
            'descriptors0_dino': tf.expand_dims(dino_descriptors0, axis=0),
            'descriptors1_dino': tf.expand_dims(dino_descriptors1, axis=0),
            'width0': tf.convert_to_tensor(
                np.expand_dims(width0, axis=0), dtype=tf.int32
            ),
            'width1': tf.convert_to_tensor(
                np.expand_dims(width1, axis=0), dtype=tf.int32
            ),
            'height0': tf.convert_to_tensor(
                np.expand_dims(height0, axis=0), dtype=tf.int32
            ),
            'height1': tf.convert_to_tensor(
                np.expand_dims(height1, axis=0), dtype=tf.int32
            ),
        }
        
        # Run OmniGlue matching on GPU 2
        print(f"> Running OmniGlue matching (GPU {og_gpu})...")
        torch.cuda.set_device(og_gpu)
        start_time = time.time()
        
        og_outputs = og_matcher.signatures['serving_default'](**inputs)
        soft_assignment = og_outputs['soft_assignment'][:, :-1, :-1]
        
        match_matrix = (
            utils.soft_assignment_to_match_matrix(soft_assignment, 1e-3)
            .numpy()
            .squeeze()
        )
        
        # Filter out any matches with 0.0 confidence keypoints.
        match_indices = np.argwhere(match_matrix)
        keep = []
        for i in range(match_indices.shape[0]):
            match = match_indices[i, :]
            if (sp_features0[2][match[0]] > 0.0) and (
                sp_features1[2][match[1]] > 0.0
            ):
                keep.append(i)
        match_indices = match_indices[keep]
        
        # Format matches in terms of keypoint locations.
        match_kp0s = []
        match_kp1s = []
        match_confidences = []
        for match in match_indices:
            match_kp0s.append(sp_features0[0][match[0], :])
            match_kp1s.append(sp_features1[0][match[1], :])
            match_confidences.append(soft_assignment[0, match[0], match[1]])
        match_kp0s = np.array(match_kp0s)
        match_kp1s = np.array(match_kp1s)
        match_confidences = np.array(match_confidences)
        
        print(f"> \tOmniGlue matching took {time.time() - start_time:.2f} seconds")
        
        # Get match info
        num_matches = match_kp0s.shape[0]
        matches = np.arange(num_matches)
        print(f"> \tFound {num_matches} matches.")
        
        # Create visualization if requested
        if visualize:
            # Use GPU 3 for visualization if available, otherwise use GPU 0
            viz_gpu = min(3, num_gpus - 1)
            print(f"> Creating visualization on GPU {viz_gpu}...")
            torch.cuda.set_device(viz_gpu)
            torch.cuda.empty_cache()
            
            viz_start = time.time()
            
            # Create the visualization
            visualization = utils.visualize_matches(
                image0, image1, match_kp0s, match_kp1s, 
                np.eye(num_matches),  # Identity matrix for matches
                show_keypoints=True,
                highlight_unmatched=True,
                title=f"{num_matches} matches",
                line_width=1,
            )
            
            print(f"> \tVisualization took {time.time() - viz_start:.2f} seconds.")
            
            # Display the visualization
            plt.figure(figsize=(15, 10))
            plt.imshow(visualization)
            plt.axis('off')
            plt.title(f"OmniGlue: {os.path.basename(image0_path)} ↔ {os.path.basename(image1_path)}")
            plt.tight_layout()
            plt.show()
            
            result = (match_kp0s, match_kp1s, match_confidences), visualization
        else:
            result = (match_kp0s, match_kp1s, match_confidences), None
        
        return result
        
    finally:
        # Clean up to release memory on all GPUs
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        gc.collect()
        print("> Memory cleaned up on all GPUs.")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python match_images_multi_gpu.py <image1_path> <image2_path> [--mode=parallel|tile|model]")
        sys.exit(1)
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    # Default mode is model parallelism
    mode = "model"
    if len(sys.argv) > 3 and sys.argv[3].startswith("--mode="):
        mode = sys.argv[3].split("=")[1]
    
    if mode == "parallel":
        # Process a single pair using data parallelism
        results = match_images_multi_gpu([(image1_path, image2_path)], max_size=1024)
        matches, visualization = results[0]
    elif mode == "tile":
        # Process using tile-based approach
        matches, visualization = match_large_images_multi_gpu(
            image1_path, image2_path, tile_size=1024, overlap=128
        )
    else:  # model
        # Process using model parallelism
        matches, visualization = match_images_model_parallel(
            image1_path, image2_path, max_size=2048
        )
    
    # Save the visualization if it exists
    if visualization is not None:
        output_path = "omniglue_matches_multi_gpu.png"
        plt.imsave(output_path, visualization)
        print(f"Saved visualization to {output_path}")
