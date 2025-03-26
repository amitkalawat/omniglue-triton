# OmniGlue Memory Optimization Guide

This guide provides strategies for optimizing memory usage when running OmniGlue, especially for large images or on systems with limited GPU memory.

## Memory Challenges with OmniGlue

OmniGlue combines three memory-intensive models:

1. **SuperPoint**: Extracts keypoints and descriptors from input images
2. **DINOv2**: A large vision foundation model that provides features to guide matching
3. **OmniGlue**: The matching model that combines outputs from the above models

The DINOv2 model in particular can consume significant GPU memory, especially with larger images.

## Single-GPU Optimization Techniques

### 1. Image Resizing

The simplest way to reduce memory usage is to resize input images:

```python
def load_and_resize_image(path, max_size=512):
    img = Image.open(path).convert('RGB')
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return np.array(img)
```

### 2. CPU-Based DINOv2

Force the memory-intensive DINOv2 model to run on CPU instead of GPU:

```python
from omniglue.src.omniglue.dino_extract import DINOExtract
original_init = DINOExtract.__init__

def cpu_init(self, cpt_path, feature_layer=11):
    # Force CPU device
    self.device = torch.device('cpu')
    original_init(self, cpt_path, feature_layer)

# Apply the monkey patch
DINOExtract.__init__ = cpu_init
```

### 3. Memory Management Settings

Configure PyTorch to use memory more efficiently:

```python
# Set environment variable for PyTorch memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Limit GPU memory usage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.5)  # Use at most 50% of available memory
```

### 4. Aggressive Garbage Collection

Clear memory frequently during processing:

```python
import gc

# Clear memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 5. Match Filtering

If there are too many matches, filter them to reduce memory usage during visualization:

```python
if num_matches > 500:
    match_threshold = 0.05  # Higher threshold to keep fewer matches
    keep_idx = []
    for i in range(match_kp0.shape[0]):
        if match_confidences[i] > match_threshold:
            keep_idx.append(i)
    
    match_kp0 = match_kp0[keep_idx]
    match_kp1 = match_kp1[keep_idx]
    match_confidences = match_confidences[keep_idx]
```

## Multi-GPU Strategies

When working with multiple GPUs (e.g., on an EC2 instance with 4 GPUs), you can leverage them to process larger images or multiple image pairs simultaneously.

### 1. Data Parallelism: Process Multiple Image Pairs

Distribute different image pairs across available GPUs:

```python
def match_images_multi_gpu(image_pairs, gpu_ids=None, max_size=1024):
    """Process multiple image pairs in parallel across multiple GPUs."""
    # See full implementation in match_images_multi_gpu.py
```

### 2. Tile-Based Processing for Large Images

Split large images into tiles and process them across multiple GPUs:

```python
def match_large_images_multi_gpu(image0_path, image1_path, gpu_ids=None, tile_size=2048, overlap=256):
    """Process large images by splitting them into tiles and processing across multiple GPUs."""
    # See full implementation in match_large_images_multi_gpu.py
```

### 3. Model Parallelism

Run different parts of the OmniGlue pipeline on different GPUs:

```python
def match_images_model_parallel(image0_path, image1_path, visualize=True, max_size=2048):
    """Process images using model parallelism across multiple GPUs.
    
    - GPU 0: SuperPoint for both images
    - GPU 1: DINOv2 for both images
    - GPU 2: OmniGlue matching
    - GPU 3: Visualization (if available)
    """
    # See full implementation in match_images_model_parallel.py
```

## Memory Usage Monitoring

To monitor GPU memory usage during processing:

```python
def check_gpu_memory(message=""):
    """Check GPU memory usage using nvidia-smi."""
    import subprocess
    import torch
    
    if not torch.cuda.is_available():
        print(f"{message} - CUDA not available")
        return
    
    try:
        # Run nvidia-smi to get memory usage
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        # Parse the output
        memory_used, memory_total = map(int, result.strip().split(','))
        
        print(f"{message} - GPU Memory: {memory_used} MB / {memory_total} MB ({memory_used/memory_total*100:.1f}%)")
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
```

## Recommended Configurations

| Image Size | Available GPU Memory | Recommended Approach |
|------------|----------------------|---------------------|
| Small (<1000px) | 4-8 GB | Single GPU with `match_images_low_memory` (max_size=512) |
| Medium (1000-2000px) | 8-16 GB | Single GPU with `match_images_low_memory` (max_size=1024) |
| Large (2000-4000px) | 16-24 GB | Single GPU with CPU-based DINOv2 |
| Very Large (4000-8000px) | 24+ GB | Multi-GPU with tile-based processing |
| Extremely Large (8000px+) | Multiple GPUs | Multi-GPU with model parallelism |

## Example Usage

```python
# For small to medium images on a single GPU with limited memory
matches, visualization = match_images_low_memory(
    './images/image1.jpg', 
    './images/image2.jpg',
    max_size=512
)

# For large images on multiple GPUs
matches, visualization = match_large_images_multi_gpu(
    './large_images/aerial1.tif', 
    './large_images/aerial2.tif',
    gpu_ids=[0, 1, 2, 3],
    tile_size=2048,
    overlap=256
)

# For processing multiple image pairs in parallel
image_pairs = [
    ('./images/pair1_img1.jpg', './images/pair1_img2.jpg'),
    ('./images/pair2_img1.jpg', './images/pair2_img2.jpg'),
    ('./images/pair3_img1.jpg', './images/pair3_img2.jpg'),
    ('./images/pair4_img1.jpg', './images/pair4_img2.jpg')
]
results = match_images_multi_gpu(image_pairs, gpu_ids=[0, 1, 2, 3])
```

## Troubleshooting

If you encounter CUDA out-of-memory errors:

1. **Reduce image size**: Try a smaller `max_size` value
2. **Force DINOv2 to CPU**: This trades speed for memory efficiency
3. **Monitor memory usage**: Use `check_gpu_memory()` to identify which part of the pipeline is consuming the most memory
4. **Split processing**: For very large images, use tile-based processing
5. **Use multiple GPUs**: Distribute the workload across available GPUs

## References

- [PyTorch Memory Management Documentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [NVIDIA Multi-GPU Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-gpu-programming)
- [OmniGlue GitHub Repository](https://github.com/google-research/omniglue)
