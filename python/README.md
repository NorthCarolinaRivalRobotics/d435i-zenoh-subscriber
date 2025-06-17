# RGB-D Feature Tracking with Zenoh and Rerun

This directory contains a refactored RGB-D feature tracking application that subscribes to Zenoh camera data and visualizes feature matching in real-time using Rerun.

## Module Structure

### `alignment_and_matching.py` (Main Application)
The main entry point that orchestrates the entire pipeline:
- Initializes Zenoh subscriber for camera data
- Coordinates feature matching between consecutive frames
- Manages the main processing loop

### `camera_config.py` (Camera Calibration)
Contains camera calibration data and utilities:
- `CameraCalibration` class with intrinsic and extrinsic parameters
- Default D435i calibration parameters
- Future extensibility for loading/saving calibration files

### `vision_utils.py` (Computer Vision Utilities)
Core computer vision and geometry functions:
- `rgb_pixel_to_xyz()`: Back-projects RGB pixels to 3D space
- `estimate_frame_transform()`: ORB feature matching and pose estimation
- Handles depth registration, feature detection, and PnP RANSAC

### `visualization.py` (Rerun Visualization)
Rerun logging and visualization utilities:
- `RerunVisualizer` class for organized logging
- Methods for RGB images, depth maps, 3D points, and camera poses
- Abstracted visualization interface

## Usage

```python
python alignment_and_matching.py
```

## Dependencies

- OpenCV (with contrib modules for RGBD)
- NumPy
- Rerun SDK
- zenoh_d435i_subscriber (custom module)

## Architecture Benefits

1. **Separation of Concerns**: Each module has a clear, focused responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Vision utilities and camera calibration can be used in other projects
4. **Maintainability**: Easier to modify and extend specific functionality
5. **Documentation**: Each module has clear interfaces and documentation 