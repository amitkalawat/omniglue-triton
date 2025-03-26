# OmniGlue Multi-GPU Processing Guide

This guide explains how to leverage multiple GPUs for processing large images with OmniGlue, specifically designed for environments like EC2 instances with multiple GPUs.

## Multi-GPU Advantages

When working with multiple GPUs (e.g., 4 GPUs with 24GB each on an EC2 instance), you can:

1. **Process larger images**: Handle images that would cause out-of-memory errors on a single GPU
2. **Increase throughput**: Process multiple image pairs simultaneously
3. **Improve performance**: Distribute different parts of the pipeline across GPUs

## Approach 1: Process Different Image Pairs on Different GPUs

This approach is ideal when you have multiple image pairs to process and want to maximize throughput.

```python
def match_images_multi_gpu(image_pairs, gpu_ids=None, max_size=1024):
    """Process multiple image pairs in parallel across multiple GPUs.
    
    Args:
        image_pairs: List of tuples [(image0_path, image1_path), ...]
        gpu_ids: List of GPU IDs to use [0, 1, 2, 3]. If None, use all available GPUs.
        max_size: Maximum dimension for images
        
    Returns:
        results: List of (matches, visualization) tuples for each image pair
    """
    import os
    import torch
    import torch.multiprocessing as mp
    from functools import partial
    
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
```

### Usage Example:

```python
image_pairs = [
    ('./images/pair1_img1.jpg', './images/pair1_img2.jpg'),
    ('./images/pair2_img1.jpg', './images/pair2_img2.jpg'),
    ('./images/pair3_img1.jpg', './images/pair3_img2.jpg'),
    ('./images/pair4_img1.jpg', './images/pair4_img2.jpg')
]
results = match_images_multi_gpu(image_pairs, gpu_ids=[0, 1, 2, 3], max_size=2048)
```

## Approach 2: Split a Single Large Image Pair Across GPUs

This approach is ideal for processing very large images that won't fit on a single GPU.

```python
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
    # See full implementation in the provided code file
```

### Usage Example:

```python
matches, visualization = match_large_images_multi_gpu(
    './large_images/aerial1.tif', 
    './large_images/aerial2.tif',
    gpu_ids=[0, 1, 2, 3],
    tile_size=2048,
    overlap=256
)
```

## Approach 3: Model Parallelism Across GPUs

This approach distributes different parts of the OmniGlue pipeline across different GPUs.

```python
def match_images_model_parallel(image0_path, image1_path, visualize=True, max_size=2048):
    """Process images using model parallelism across multiple GPUs.
    
    This function distributes different parts of the OmniGlue pipeline across different GPUs:
    - GPU 0: SuperPoint for both images
    - GPU 1: DINOv2 for both images
    - GPU 2: OmniGlue matching
    - GPU 3: Visualization (if available)
    """
    # See full implementation in the provided code file
```

### Usage Example:

```python
matches, visualization = match_images_model_parallel(
    './large_images/image1.jpg',
    './large_images/image2.jpg',
    max_size=4096  # Can process much larger images with multiple GPUs
)
```

## Performance Considerations

### 1. GPU Interconnect Bandwidth

For model parallelism, the bandwidth between GPUs is crucial. EC2 instances with NVLink or high-bandwidth PCIe connections (like p3.8xlarge or p4d.24xlarge) will perform better when sharing data between GPUs.

### 2. Memory Management

Each GPU has its own memory, so you can process images up to the memory capacity of each GPU (e.g., 24GB per GPU).

### 3. Load Balancing

When processing multiple image pairs or tiles, distribute the workload evenly across GPUs. The provided code uses round-robin assignment, but you could implement more sophisticated load balancing based on image sizes or GPU capabilities.

### 4. Monitoring GPU Usage

Monitor GPU usage during processing:

```bash
watch -n 0.5 nvidia-smi
```

## EC2 Instance Recommendations

For multi-GPU OmniGlue processing, consider these EC2 instance types:

| Instance Type | GPUs | GPU Memory | Best For |
|---------------|------|------------|----------|
| p3.8xlarge    | 4x V100 | 16GB each | Medium to large images |
| p3.16xlarge   | 8x V100 | 16GB each | Multiple image pairs or very large images |
| p4d.24xlarge  | 8x A100 | 40GB each | Extremely large images or high throughput |
| g5.12xlarge   | 4x A10G | 24GB each | Cost-effective for medium to large images |

## Example Workflow for Large Aerial Image Matching

```python
import os
import matplotlib.pyplot as plt
from multi_gpu_processing import match_large_images_multi_gpu

# Process large aerial images
aerial_image1 = './aerial_images/region1_2022.tif'  # 10000x10000 pixels
aerial_image2 = './aerial_images/region1_2023.tif'  # 10000x10000 pixels

# Use all 4 GPUs with tile-based processing
matches, visualization = match_large_images_multi_gpu(
    aerial_image1, 
    aerial_image2,
    gpu_ids=[0, 1, 2, 3],
    tile_size=2048,
    overlap=256
)

# Save the visualization
plt.figure(figsize=(20, 15))
plt.imshow(visualization)
plt.axis('off')
plt.title(f"OmniGlue: Aerial Image Matching ({len(matches[0])} matches)")
plt.savefig('aerial_matches.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Found {len(matches[0])} matches between aerial images")
print(f"Visualization saved to aerial_matches.png")
```

## Conclusion

By leveraging multiple GPUs, you can process much larger images with OmniGlue than would be possible on a single GPU, or process multiple image pairs in parallel for higher throughput. Choose the approach that best fits your specific use case and available hardware.
