#!/usr/bin/env python3
"""
Test script for the recording and playback system.
"""
from __future__ import annotations

import sys
import os
import time

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_source import create_data_source, LiveCameraDataSource, RecordingCameraDataSource, PlaybackCameraDataSource
    from frame_reconstruction import PlaybackFrameData
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_data_source_creation():
    """Test creating different types of data sources."""
    print("\n🧪 Testing data source creation...")
    
    # Test live data source
    try:
        live_source = create_data_source()
        print("✅ Live data source created")
        assert isinstance(live_source, LiveCameraDataSource)
    except Exception as e:
        print(f"❌ Failed to create live data source: {e}")
    
    # Test recording data source
    try:
        recording_source = create_data_source(record=True, recording_file="recordings/test.pkl.gz")
        print("✅ Recording data source created")
        assert isinstance(recording_source, RecordingCameraDataSource)
    except Exception as e:
        print(f"❌ Failed to create recording data source: {e}")
    
    # Test playback data source (will fail because file doesn't exist, but that's expected)
    try:
        playback_source = create_data_source(playback_file="nonexistent.pkl.gz")
        print("❌ Playback data source should have failed but didn't")
    except FileNotFoundError:
        print("✅ Playback data source correctly failed for missing file")
    except Exception as e:
        print(f"❌ Unexpected error creating playback data source: {e}")


def test_frame_reconstruction():
    """Test frame reconstruction from recorded data."""
    print("\n🧪 Testing frame reconstruction...")
    
    # Create mock recorded frame data with correct dimensions
    width, height = 640, 480
    total_pixels = width * height
    
    mock_frame_record = {
        'timestamp': 1.5,
        'frame_count': 42,
        'rgb': {
            'data': b'\x00\x01\x02\x03' * (width * height * 3 // 4),  # Mock RGB data (3 bytes per pixel)
            'timestamp': 1234567890.5,
            'width': width,
            'height': height
        },
        'depth': {
            'raw_data': [i % 65536 for i in range(total_pixels)],  # Mock depth data with valid uint16 values (as list, like saved data)
            'timestamp': 1234567890.5,
            'width': width,
            'height': height
        },
        'motion': {
            'gyro': [0.1, 0.2, 0.3],
            'accel': [9.8, 0.1, 0.2],
            'timestamp': 1234567890.5
        }
    }
    
    try:
        playback_frame = PlaybackFrameData(mock_frame_record)
        
        # Test frame properties
        assert playback_frame.frame_count == 42
        assert playback_frame.timestamp == 1.5
        
        # Test RGB frame
        rgb_frame = playback_frame.rgb
        assert rgb_frame is not None
        assert rgb_frame.width == width
        assert rgb_frame.height == height
        rgb_data = rgb_frame.get_data()
        assert len(rgb_data) > 0
        
        # Test depth frame
        depth_frame = playback_frame.depth
        assert depth_frame is not None
        assert depth_frame.width == width
        assert depth_frame.height == height
        depth_2d = depth_frame.get_data_2d()
        assert depth_2d.shape == (height, width)
        
        # Test motion frame
        motion_frame = playback_frame.motion
        assert motion_frame is not None
        assert len(motion_frame.gyro) == 3
        assert len(motion_frame.accel) == 3
        
        print("✅ Frame reconstruction tests passed")
        
    except Exception as e:
        print(f"❌ Frame reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()


def test_empty_frame():
    """Test handling of empty frames."""
    print("\n🧪 Testing empty frame handling...")
    
    empty_frame_record = {
        'timestamp': 0,
        'frame_count': 0,
        'rgb': None,
        'depth': None,
        'motion': None
    }
    
    try:
        empty_frame = PlaybackFrameData(empty_frame_record)
        
        assert empty_frame.frame_count == 0
        assert empty_frame.rgb is None
        assert empty_frame.depth is None
        assert empty_frame.motion is None
        
        print("✅ Empty frame handling tests passed")
        
    except Exception as e:
        print(f"❌ Empty frame test failed: {e}")


def main():
    """Run all tests."""
    print("🔬 Testing Camera Recording/Playback System")
    print("=" * 50)
    
    test_data_source_creation()
    test_frame_reconstruction()
    test_empty_frame()
    
    print("\n✨ All tests completed!")
    print("\nTo test the full system:")
    print("1. Record some data: python alignment_minimal.py --record")
    print("2. Play it back: python alignment_minimal.py --playback recordings/<filename>.pkl.gz")


if __name__ == "__main__":
    main() 