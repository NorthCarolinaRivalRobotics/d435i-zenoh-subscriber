# Camera Recording/Playback Integration Guide

This guide shows how to add recording and playback functionality to any script that uses the Zenoh D435i subscriber.

## 🚀 Quick Integration (3 steps)

### Before (Original Code)
```python
#!/usr/bin/env python3
import zenoh_d435i_subscriber as zd435i

def main():
    # Original subscriber setup
    sub = zd435i.ZenohD435iSubscriber()
    sub.connect()
    sub.start_subscribing()
    
    while sub.is_running():
        frame_data = sub.get_latest_frames()
        # ... process frame_data ...
    
    sub.stop()

if __name__ == "__main__":
    main()
```

### After (With Recording/Playback)
```python
#!/usr/bin/env python3
import sys
import os

# Step 1: Add import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_data_manager import setup_camera_manager

def main():
    # Step 2: Replace subscriber setup
    camera_manager, args = setup_camera_manager("My application")
    camera_manager.connect()
    camera_manager.start_subscribing()
    
    # Step 3: Use camera_manager instead of sub (API is identical!)
    while camera_manager.is_running():
        frame_data = camera_manager.get_latest_frames()
        # ... process frame_data exactly the same way ...
    
    camera_manager.stop()

if __name__ == "__main__":
    main()
```

**That's it!** Your script now supports:
- `python script.py` (live mode)
- `python script.py --record` (recording mode)
- `python script.py --playback recordings/data.pkl.gz` (playback mode)

## 🔧 Advanced Integration Options

### Option 1: Manual Setup (More Control)
```python
from camera_data_manager import CameraDataManager, add_camera_args
import argparse

def main():
    # Manual argument parsing
    parser = argparse.ArgumentParser(description="My camera application")
    add_camera_args(parser)
    # Add your own arguments here
    parser.add_argument('--my-option', help='My custom option')
    args = parser.parse_args()
    
    # Create manager manually
    camera_manager = CameraDataManager(
        record=args.record,
        playback_file=args.playback,
        recording_file=getattr(args, 'recording_file', None),
        loop=getattr(args, 'loop', True),
        realtime=not getattr(args, 'no_realtime', False)
    )
    
    print(f"Running in {camera_manager.get_mode()} mode")
    
    # Rest of your code...
```

### Option 2: Programmatic Setup (No Command Line)
```python
from camera_data_manager import CameraDataManager

def main():
    # Create manager programmatically
    # For recording:
    camera_manager = CameraDataManager(record=True, recording_file="my_session.pkl.gz")
    
    # For playback:
    # camera_manager = CameraDataManager(playback_file="recordings/data.pkl.gz")
    
    # For live (default):
    # camera_manager = CameraDataManager()
    
    # Rest of your code...
```

## 📝 Real Examples

### Example 1: Adding to alignment_minimal.py
**Before:**
```python
# Setup Zenoh subscriber
sub = zd435i.ZenohD435iSubscriber()
sub.connect()
sub.start_subscribing()

while sub.is_running():
    fd = sub.get_latest_frames()
    # ... process frames ...
```

**After:**
```python
# Set up camera manager with command line arguments
camera_manager, args = setup_camera_manager("Minimal latency test")
camera_manager.connect()
camera_manager.start_subscribing()

while camera_manager.is_running():
    fd = camera_manager.get_latest_frames()
    # ... process frames exactly the same way ...
```

### Example 2: Adding to alignment_and_matching.py
**Before:**
```python
# Setup Zenoh subscriber
sub = zd435i.ZenohD435iSubscriber()
sub.connect()
sub.start_subscribing()

while sub.is_running():
    fd = sub.get_latest_frames()
    # ... feature matching and pose estimation ...
```

**After:**
```python
# Set up camera manager with command line arguments
camera_manager, args = setup_camera_manager("RGB-D feature tracking")
camera_manager.connect()
camera_manager.start_subscribing()

while camera_manager.is_running():
    fd = camera_manager.get_latest_frames()
    # ... feature matching and pose estimation (no changes needed) ...
```

## 🧪 Testing Your Integration

### 1. Run the Test Suite
```bash
python test_camera_data_manager.py
```

### 2. Test Your Script
```bash
# Test live mode (should work as before)
python your_script.py

# Test recording mode
python your_script.py --record

# Test playback mode (after recording some data)
python your_script.py --playback recordings/camera_data_*.pkl.gz
```

## 🔍 API Compatibility

The `CameraDataManager` provides the exact same interface as `ZenohD435iSubscriber`:

| Method | Description | Compatibility |
|--------|-------------|---------------|
| `connect()` | Connect to data source | ✅ 100% |
| `start_subscribing()` | Start data flow | ✅ 100% |
| `get_latest_frames()` | Get frame data | ✅ 100% |
| `is_running()` | Check if active | ✅ 100% |
| `stop()` | Stop data flow | ✅ 100% |

**Frame Data Compatibility:**
- RGB data: `frame_data.rgb.get_data()` ✅
- Depth data: `frame_data.depth.get_data_2d()` ✅
- Motion data: `frame_data.motion.gyro`, `frame_data.motion.accel` ✅
- Frame count: `frame_data.frame_count` ✅

## 🎛️ Command Line Options

After integration, your script automatically supports these options:

| Option | Description | Example |
|--------|-------------|---------|
| `--record` | Record while running | `python script.py --record` |
| `--playback FILE` | Play recorded data | `python script.py --playback data.pkl.gz` |
| `--recording-file FILE` | Specify recording file | `python script.py --record --recording-file session1.pkl.gz` |
| `--loop` / `--no-loop` | Control playback looping | `python script.py --playback data.pkl.gz --no-loop` |
| `--no-realtime` | Fast playback | `python script.py --playback data.pkl.gz --no-realtime` |

## 🐛 Troubleshooting

### Import Errors
```python
# Make sure you add the path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_data_manager import setup_camera_manager
```

### File Not Found Errors
```bash
# Check recordings directory
ls -la recordings/

# Use full path if needed
python script.py --playback /full/path/to/recording.pkl.gz
```

### Performance Issues
```bash
# Use fast playback for data processing
python script.py --playback data.pkl.gz --no-realtime

# Check file sizes
du -h recordings/*.pkl.gz
```

## 📊 Migration Checklist

- [ ] Add import: `from camera_data_manager import setup_camera_manager`
- [ ] Replace subscriber creation with `camera_manager, args = setup_camera_manager("App name")`
- [ ] Change `sub.` to `camera_manager.` in all method calls
- [ ] Test live mode: `python script.py`
- [ ] Test recording mode: `python script.py --record`
- [ ] Test playback mode: `python script.py --playback recordings/file.pkl.gz`
- [ ] Update any status messages to use `camera_manager.get_status_string()`

## 🎯 Benefits After Integration

✅ **Zero Code Changes**: Your processing logic stays exactly the same  
✅ **Automatic CLI**: Get recording/playback flags for free  
✅ **Debugging**: Record problematic sessions and replay them  
✅ **Testing**: Test algorithms on consistent data  
✅ **Demos**: Show your work without live camera  
✅ **Development**: Work offline with recorded data  

## 🚀 Next Steps

1. **Start Simple**: Use the 3-step quick integration
2. **Test Thoroughly**: Run all three modes (live, record, playback)
3. **Customize**: Add your own command line options as needed
4. **Share**: Your recordings can be shared with other developers

Your script is now a professional tool with recording and playback capabilities! 🎉 