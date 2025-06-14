# Zenoh D435i Subscriber

A high-performance Python package built with PyO3 and Maturin for subscribing to RealSense D435i camera data via Zenoh.

## Features

- 🚀 **High Performance**: Built with Rust for maximum speed
- 📸 **Multi-Stream Support**: RGB, Depth, and Motion data
- 🔄 **Real-time Processing**: ~30 FPS data streaming
- 🐍 **Python Integration**: Easy-to-use Python API
- 📊 **Data Visualization**: Built-in depth image plotting
- 🌐 **Zenoh Networking**: Distributed data sharing

## Quick Start

### Prerequisites

- Python 3.8+
- Rust (latest stable)
- Maturin for Python-Rust integration

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd zenoh-d435i-subscriber
   ```

2. **Install development dependencies:**
   ```bash
   make dev-deps
   ```

3. **Build and install:**
   ```bash
   make install
   ```

4. **Test the installation:**
   ```bash
   make test
   ```

5. **Run the example:**
   ```bash
   make example
   ```

### Build Commands

- `make help` - Show all available commands
- `make build` - Build in development mode
- `make build-release` - Build in release mode (optimized)
- `make install` - Quick install (build + install)
- `make test` - Test the installation
- `make example` - Run the example
- `make clean` - Clean build artifacts
- `make dev-deps` - Install development dependencies

### Manual Build (Alternative)

If you prefer to build manually:

```bash
# Development build
./build.sh

# Release build
./build.sh release
```

## Usage

```python
import zenoh_d435i_subscriber as zd435i
import time

# Create subscriber
subscriber = zd435i.ZenohD435iSubscriber()

# Set up camera intrinsics (optional)
depth_intrinsics = zd435i.PyIntrinsics(
    width=640, height=480,
    fx=387.31454, fy=387.31454,
    ppx=322.1206, ppy=236.50139
)
subscriber.set_depth_intrinsics(depth_intrinsics)

# Connect and start subscribing
subscriber.connect()
subscriber.start_subscribing()

# Poll for data
while subscriber.is_running():
    frame_data = subscriber.get_latest_frames()
    
    if frame_data.frame_count > 0:
        print(f"📊 {frame_data}")
        
        # Access RGB data
        if frame_data.rgb:
            rgb_bytes = frame_data.rgb.get_data()
            print(f"RGB: {len(rgb_bytes)} bytes")
        
        # Access depth data  
        if frame_data.depth:
            depth_array = frame_data.depth.get_data_2d()
            print(f"Depth: {depth_array.shape}")
        
        # Access motion data
        if frame_data.motion:
            print(f"Gyro: {frame_data.motion.gyro}")
            print(f"Accel: {frame_data.motion.accel}")
    
    time.sleep(0.1)
```

## API Reference

### Classes

- **`ZenohD435iSubscriber`**: Main subscriber class
- **`PyIntrinsics`**: Camera intrinsics data
- **`PyExtrinsics`**: Camera extrinsics data
- **`PyFrameData`**: Combined frame data container
- **`PyRgbFrame`**: RGB frame data
- **`PyDepthFrame`**: Depth frame data with NumPy integration
- **`PyMotionFrame`**: IMU motion data

### Key Methods

- `connect()`: Connect to Zenoh network
- `start_subscribing()`: Begin data subscription
- `get_latest_frames()`: Get most recent frame data (non-blocking)
- `get_stats()`: Get frame statistics
- `stop()`: Stop subscription
- `is_running()`: Check if subscriber is active

## Troubleshooting

### Build Issues

If you encounter build problems:

1. **Clean and rebuild:**
   ```bash
   make clean
   make install
   ```

2. **Check Python environment:**
   ```bash
   which python
   python --version
   ```

3. **Verify dependencies:**
   ```bash
   make dev-deps
   ```

### Import Issues

If the module can't be imported:

1. **Verify installation:**
   ```bash
   make test
   ```

2. **Check pip list:**
   ```bash
   pip list | grep zenoh
   ```

3. **Reinstall:**
   ```bash
   make clean
   make install
   ```

## Development

### Build Process

The build system has been optimized for reliability:

1. **Automatic Python detection**: Uses the currently active Python interpreter
2. **Clean builds**: Automatically cleans previous builds
3. **Wheel-based installation**: Builds wheel then installs via pip
4. **Verification**: Tests import after installation

### Architecture

- **Rust Core**: High-performance data processing and Zenoh integration
- **PyO3 Bindings**: Seamless Python-Rust interop
- **NumPy Integration**: Efficient array data sharing
- **Tokio Runtime**: Async data streaming

## Dependencies

### Python
- numpy
- matplotlib (for examples)
- opencv-python (optional, for development)
- pillow (optional, for development)

### Rust
- pyo3: Python integration
- zenoh: Distributed data sharing
- tokio: Async runtime
- numpy: NumPy integration
- nalgebra: Linear algebra
- turbojpeg: JPEG compression
- zstd: Compression

## Performance

- **Throughput**: ~30 FPS for combined RGB+Depth+Motion streams
- **Latency**: Sub-millisecond data access via polling API
- **Memory**: Efficient zero-copy data sharing where possible
- **CPU**: Optimized Rust core with minimal Python overhead

## License

This project is licensed under the MIT License - see the LICENSE file for details. 