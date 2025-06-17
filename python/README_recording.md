# Camera Data Recording and Playback System

This system provides the ability to record camera data from the Zenoh D435i subscriber and play it back later, while maintaining the exact same API as the live data stream.

## Features

- **🔴 Recording Mode**: Records live camera data to compressed files
- **▶️ Playback Mode**: Plays back recorded data with timing control
- **🎯 Same API**: Drop-in replacement for live data source
- **💾 Compression**: Uses gzip compression for efficient storage
- **⏱️ Timing Control**: Real-time or fast playback options

## Architecture

The system uses an abstract base class `CameraDataSource` with three implementations:

1. **`LiveCameraDataSource`**: Wraps the original Zenoh subscriber
2. **`RecordingCameraDataSource`**: Records data while providing live stream
3. **`PlaybackCameraDataSource`**: Plays back recorded data

## Usage

### Live Mode (Default)
```bash
python alignment_minimal.py
```

### Recording Mode
```bash
# Record with auto-generated filename
python alignment_minimal.py --record

# Record to specific file
python alignment_minimal.py --record --recording-file recordings/my_session.pkl.gz
```

### Playback Mode
```bash
# Play back recorded data
python alignment_minimal.py --playback recordings/camera_data_20231201_120000.pkl.gz

# Play back without looping
python alignment_minimal.py --playback recordings/data.pkl.gz --no-loop

# Play back as fast as possible (not real-time)
python alignment_minimal.py --playback recordings/data.pkl.gz --no-realtime
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--record` | Enable recording mode |
| `--playback FILE` | Play back from recorded file |
| `--recording-file FILE` | Specify recording filename |
| `--loop` / `--no-loop` | Control playback looping (default: loop) |
| `--no-realtime` | Play back as fast as possible |

## File Format

Recorded files use the following format:
- **Compression**: gzip compressed pickle files
- **Extension**: `.pkl.gz`
- **Content**: Sequence of frame records with RGB, depth, and motion data
- **Timestamps**: Relative timestamps for accurate playback timing

### Frame Record Structure
```python
{
    'timestamp': float,        # Relative timestamp from recording start
    'frame_count': int,        # Frame sequence number
    'rgb': {                   # RGB frame data (if available)
        'data': bytes,         # Compressed JPEG data
        'timestamp': float,    # Original timestamp
        'width': int,         # Image width
        'height': int         # Image height
    },
    'depth': {                # Depth frame data (if available)
        'raw_data': List[int], # Raw uint16 depth values
        'timestamp': float,    # Original timestamp
        'width': int,         # Image width
        'height': int         # Image height
    },
    'motion': {               # Motion data (if available)
        'gyro': List[float],  # Gyroscope data [x, y, z]
        'accel': List[float], # Accelerometer data [x, y, z]
        'timestamp': float    # Original timestamp
    }
}
```

## API Compatibility

The recording and playback system maintains full API compatibility with the original live data source:

```python
# Original code (still works)
sub = zd435i.ZenohD435iSubscriber()
sub.connect()
sub.start_subscribing()
frame_data = sub.get_latest_frames()

# New abstracted code (works with live, recording, or playback)
data_source = create_data_source(record=True)  # or playback_file="..."
data_source.connect()
data_source.start_subscribing()
frame_data = data_source.get_latest_frames()
```

## Storage Requirements

Recording storage depends on frame rate and data content:
- **RGB data**: ~50-200 KB per frame (JPEG compressed)
- **Depth data**: ~600 KB per frame (raw uint16 values)
- **Motion data**: ~100 bytes per frame
- **Total**: ~650-800 KB per frame
- **Example**: 30 FPS for 10 minutes ≈ 12-14 GB

## Playback Features

### Real-time Playback
- Maintains original timing between frames
- Suitable for realistic simulation
- Default behavior

### Fast Playback
- Plays frames as fast as possible
- Useful for data processing pipelines
- Enable with `--no-realtime`

### Looping
- Automatically restarts when reaching end
- Useful for continuous testing
- Default behavior (disable with `--no-loop`)

## Testing

Run the test suite to verify functionality:

```bash
python test_recording_system.py
```

## File Management

### Default Recording Location
- Directory: `recordings/`
- Filename format: `camera_data_YYYYMMDD_HHMMSS.pkl.gz`
- Example: `camera_data_20231201_143022.pkl.gz`

### Manual File Management
```bash
# List recordings
ls -lh recordings/

# Check file size
du -h recordings/camera_data_*.pkl.gz

# Clean up old recordings
rm recordings/camera_data_202311*.pkl.gz
```

## Performance Notes

### Recording Performance
- Minimal impact on live visualization
- Background compression and writing
- Real-time status updates every 100 frames

### Playback Performance
- Fast loading of recorded data
- Lazy frame reconstruction
- Memory efficient with large files

## Troubleshooting

### Common Issues

**ImportError on frame_reconstruction**
```bash
# Ensure you're in the python directory
cd python
python alignment_minimal.py --record
```

**File not found errors**
```bash
# Check file path
ls -la recordings/
python alignment_minimal.py --playback recordings/your_file.pkl.gz
```

**Memory issues with large recordings**
- Use `--no-realtime` for faster processing
- Process recordings in chunks if needed
- Monitor system memory usage

### Debug Mode
Enable verbose output by modifying the data source creation in your script:
```python
# Add debug prints in data_source.py for troubleshooting
```

## Integration with Existing Code

To add recording/playback to existing scripts:

1. **Import the module**:
   ```python
   from data_source import create_data_source
   ```

2. **Replace subscriber creation**:
   ```python
   # Old
   sub = zd435i.ZenohD435iSubscriber()
   
   # New
   sub = create_data_source(record=args.record, playback_file=args.playback)
   ```

3. **Add command line parsing**:
   ```python
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--record', action='store_true')
   parser.add_argument('--playback', type=str)
   args = parser.parse_args()
   ```

The rest of your code remains unchanged! 