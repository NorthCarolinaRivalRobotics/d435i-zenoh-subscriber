# Sensor Fusion System

This directory contains a comprehensive sensor fusion system that combines IMU (Inertial Measurement Unit) data with RGB-D visual odometry using GTSAM (Georgia Tech Smoothing and Mapping) for optimal state estimation.

## Overview

The fusion system integrates:
- **IMU Data**: Gyroscope and accelerometer measurements at ~100Hz
- **Visual Odometry**: Multi-frame RGB-D feature tracking at ~10Hz
- **GTSAM Backend**: Factor graph optimization for robust sensor fusion
- **Rerun Visualization**: Real-time debugging and visualization

## Key Components

### Core Files

1. **`fusion_pose_estimate.py`** - Main fusion system
   - Combines IMU and visual odometry measurements
   - Uses GTSAM for factor graph optimization
   - Provides real-time visualization with Rerun

2. **`fusion_backend.py`** - GTSAM optimization backend
   - IMU preintegration
   - RGB-D pose factors
   - Sliding window optimization

3. **`fusion_test_utils.py`** - Testing utilities
   - Synthetic trajectory generation
   - Mock sensors for testing
   - Validation tools

### Supporting Components

- **`gyro_angle_estimate.py`** - Standalone gyroscope integration
- **`alignment_and_matching.py`** - Visual odometry with multi-frame tracking
- **`visualization.py`** - Extended Rerun visualization utilities
- **`run_fusion.py`** - Convenience runner script

## Architecture

The system follows a modular architecture with clear abstractions:

```
┌─────────────────┐     ┌──────────────────┐
│   IMU Sensor    │     │ Visual Odometry  │
│   Interface     │     │     Sensor       │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │  IMUMeasurement       │  VisualOdometryMeasurement
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────┴──────┐
              │   Sensor    │
              │   Fusion    │
              └──────┬──────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    ┌────┴─────┐          ┌──────┴──────┐
    │  GTSAM   │          │   Rerun     │
    │ Backend  │          │Visualization│
    └──────────┘          └─────────────┘
```

## Usage

### Quick Start

Use the convenience runner script:

```bash
# Run live fusion with real sensor data
python run_fusion.py live

# Run with synthetic test data
python run_fusion.py test

# Calibrate IMU
python run_fusion.py calibrate

# Run individual components
python run_fusion.py gyro-only
python run_fusion.py vo-only
```

### Recording and Playback

```bash
# Record sensor data
python run_fusion.py live --record

# Playback recorded data
python run_fusion.py live --playback recordings/camera_data_20231201_120000.pkl.gz
```

### Direct Execution

```bash
# Run fusion system directly
python fusion_pose_estimate.py

# Run with recording
python fusion_pose_estimate.py --record

# Run synthetic tests
python fusion_test_utils.py
```

## Key Features

### 1. Sensor Abstraction
- Clean interface for different sensor types
- Easy to extend with new sensors
- Supports both real and synthetic data

### 2. Multi-rate Sensor Fusion
- Handles different sensor frequencies gracefully
- IMU at ~100Hz, Visual Odometry at ~10Hz
- Fusion updates at configurable rate (default 100Hz)

### 3. Robust Optimization
- GTSAM factor graph for optimal estimation
- IMU preintegration for efficiency
- Sliding window for bounded computation

### 4. Comprehensive Visualization
- Real-time 3D visualization with Rerun
- IMU data plots
- Visual odometry matches
- Fused trajectory
- Uncertainty visualization
- Sensor status monitoring

### 5. Testing Support
- Synthetic trajectory generation
- Circular and sinusoidal motion patterns
- Configurable noise models
- Ground truth comparison

## Configuration

### IMU Configuration
- Calibration file: `imu_calibration.json`
- Coordinate transform: Y↔Z axis swap for D435i
- Gyroscope noise: 0.01 rad/s
- Accelerometer noise: 0.1 m/s²

### Visual Odometry Configuration
- Multi-frame tracking: 4 frames
- Minimum inliers: 10
- Feature detector: ORB
- Matcher: FLANN-based

### Fusion Configuration
- Window size: 100 states
- Gravity: 9.81 m/s²
- Update rate: 100 Hz

## Visualization Guide

The Rerun visualization shows:

1. **3D View**
   - World coordinate frame
   - Current robot pose
   - Trajectory history
   - Feature matches
   - Coordinate frames (IMU, Camera)

2. **Time Series Plots**
   - IMU gyroscope (X, Y, Z)
   - IMU accelerometer (X, Y, Z)
   - Visual odometry translation
   - Sensor rates
   - Fusion state

3. **Status Information**
   - Sensor rates (Hz)
   - Latencies (ms)
   - Number of features tracked
   - Optimization metrics

## Troubleshooting

### No IMU Data
1. Check Zenoh connection
2. Verify IMU is publishing
3. Run calibration if needed

### Poor Visual Odometry
1. Ensure good lighting
2. Avoid textureless surfaces
3. Check for motion blur
4. Increase minimum features

### Drift Issues
1. Recalibrate IMU
2. Check coordinate transforms
3. Tune noise parameters
4. Verify sensor synchronization

## Development

### Adding New Sensors
1. Implement `SensorInterface`
2. Create measurement dataclass
3. Add to fusion system
4. Update visualization

### Testing
```python
# Run unit tests
pytest test_fusion.py

# Run integration tests
python fusion_test_utils.py
```

### Performance Profiling
The system includes built-in profiling:
- Per-function timing
- Memory usage tracking
- Bottleneck identification

## Dependencies

- GTSAM (Python bindings)
- Rerun SDK
- OpenCV
- NumPy
- Zenoh D435i subscriber (Rust module)

## Future Improvements

1. **Loop Closure** - Detect and handle revisited locations
2. **Map Building** - Create persistent 3D maps
3. **Multi-camera Support** - Fuse multiple camera streams
4. **GPS Integration** - Add global positioning
5. **Learning-based Features** - Use deep learning for robust features 