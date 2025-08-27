# Data Conversion Scripts

This directory contains scripts to convert robotics datasets from various formats to the LeRobot v2.1 format.

## Available Conversion Scripts

### `hdf5_to_lerobot.py`
Converts datasets from HDF5 format to LeRobot format. Designed for datasets where each HDF5 file represents a single episode with the structure:
- `/data/demo_0/action` - Actions taken at each step
- `/data/demo_0/observations/rgb` - RGB image observations  
- `/data/demo_0/abs_joint_pos` - Absolute joint positions
- `/data/demo_0/timestep` - Timestamps for each data point

### `zarr_to_lerobot.py`
Converts datasets from Zarr format to LeRobot format. Handles single Zarr stores containing multiple episodes with episode boundaries defined by an `episode_ends` array.

### `dvrk_zarr_to_lerobot.py`
Specialized conversion script for DVRK (da Vinci Research Kit) datasets. Processes directory structures with multiple cameras (endoscope, wrist), handles recovery demonstrations, and includes surgical tool metadata.

### `custom_lerobot_split.py`
Demonstrates how to create custom dataset splits including recovery and failure examples, useful for training robust policies in safety-critical applications.

---

## Performance Optimization

### Video Encoding Parameters

LeRobot dataset creation supports several parameters that can significantly improve conversion performance for large datasets:

#### `image_writer_processes` and `image_writer_threads`
These parameters control parallel video encoding:
- **`image_writer_processes`**: Number of parallel processes for video encoding
- **`image_writer_threads`**: Number of threads per encoding process

**Performance Impact:**
- Default (no parallelization): ~947 seconds for small dataset
- Optimized (15 threads, 10 processes): ~316 seconds (**3x faster**)

**Recommended Values:**
- `image_writer_processes=10-16` (adjust based on CPU cores)
- `image_writer_threads=15-20` (balance between throughput and memory usage)

#### `tolerance_s`
Time tolerance for data synchronization between different sensors (default: 0.1 seconds). Adjust based on your system's timing precision requirements.

#### `batch_encoding_size` (Advanced)
Controls how many episodes are batched together before video encoding:
- **Benefits**: Further performance improvement (~8% faster)
- **Caveat**: Episodes in incomplete batches remain as individual images rather than MP4 videos
- **Recommendation**: Use only for large datasets where batch size divides evenly into total episode count

### Optimal Configuration Example

```python
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    use_videos=True,
    robot_type="your_robot",
    fps=30,
    features={...},
    # Performance optimization parameters
    image_writer_processes=16,
    image_writer_threads=20,
    tolerance_s=0.1,
    # batch_encoding_size=12,  # Use with caution - see notes above
)
```