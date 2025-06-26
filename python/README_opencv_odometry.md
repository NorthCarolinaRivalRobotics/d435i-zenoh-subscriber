# OpenCV RGB-D Odometry Integration

This implementation provides a drop-in replacement for the custom visual odometry using OpenCV's built-in RGB-D odometry algorithms.

## Files Created

- `opencv_rgbd_odometry.py` - OpenCV RGB-D odometry sensor implementation
- `fusion_opencv_demo.py` - Demo script showing integration with fusion system

## Usage

### Basic Usage

Run the fusion system with OpenCV RGB-D odometry:

```bash
# Using the run script
python run_fusion.py opencv

# Or directly
python fusion_opencv_demo.py
```

### Algorithm Options

OpenCV provides several RGB-D odometry algorithms:

1. **RgbdOdometry** (default) - Combined RGB and depth features
   ```bash
   python run_fusion.py opencv --algorithm RgbdOdometry
   ```

2. **ICPOdometry** - Iterative Closest Point using only depth
   ```bash
   python run_fusion.py opencv --algorithm ICPOdometry
   ```

3. **RgbdICPOdometry** - Hybrid approach combining both
   ```bash
   python run_fusion.py opencv --algorithm RgbdICPOdometry
   ```

4. **FastICPOdometry** - Faster ICP variant
   ```bash
   python run_fusion.py opencv --algorithm FastICPOdometry
   ```

### Advanced Mode

Enable advanced features with validation and parameter tuning:

```bash
python run_fusion.py opencv --algorithm RgbdICPOdometry --advanced
```

## Key Features

### Depth Registration
The implementation automatically registers depth images to the RGB camera frame using OpenCV's `registerDepth`:
- Handles camera intrinsics and extrinsics
- Performs depth dilation to fill small holes
- Converts depth to meters with proper scaling

### Coordinate Transforms
- Maintains compatibility with existing fusion system
- Applies camera-to-robot coordinate transforms
- Preserves gyroscope integration for rotation

### Error Handling
- Validates transformation magnitudes
- Filters outlier measurements
- Provides fallback for failed computations

## Integration with Existing System

The OpenCV odometry sensor implements the same `SensorInterface` as the original visual odometry, making it a drop-in replacement:

```python
# Original
self.vo_system = VisualOdometrySensor(camera_manager, camera_cal)

# OpenCV replacement
self.vo_system = create_opencv_visual_odometry(camera_manager, camera_cal)
```

## Performance Comparison

| Algorithm | Speed | Accuracy | Robustness |
|-----------|-------|----------|------------|
| RgbdOdometry | Medium | High | Good |
| ICPOdometry | Fast | Medium | Good for geometry |
| RgbdICPOdometry | Slow | Highest | Best |
| FastICPOdometry | Fastest | Medium | Good |

## Troubleshooting

### Common Issues

1. **"OpenCV RGB-D odometry failed"**
   - Insufficient features/depth data
   - Try different algorithm or adjust depth range

2. **Large transformation warnings**
   - Enable advanced mode for validation
   - Adjust `max_translation` and `max_rotation` parameters

3. **Low frame rate**
   - Try FastICPOdometry for better speed
   - Reduce image resolution if needed

### Debug Output

The implementation provides detailed debug information:
- Translation magnitudes
- Algorithm being used
- Failure reasons

## Extending the Implementation

To add custom parameters or new algorithms:

1. Modify `OpenCVRgbdOdometryAdvanced` class
2. Add new algorithm initialization in `__init__`
3. Update validation logic in `get_measurement`

## Dependencies

Requires OpenCV with contrib modules:
```bash
pip install opencv-contrib-python
```

Make sure your OpenCV build includes the `rgbd` module. 