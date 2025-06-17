#!/usr/bin/env python3
"""
Debug script to understand playback behavior.
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import sys
import os

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_data_manager import setup_camera_manager
import camera_data_manager


def main():
    """Debug playback to see what's happening."""
    
    # Check if we have a recording file
    import glob
    recording_files = glob.glob("recordings/*.pkl.gz")
    if not recording_files:
        print("No recording files found. Creating a mock recording for testing...")
        # Create a simple test recording
        create_test_recording()
        recording_files = glob.glob("recordings/*.pkl.gz")
    
    if recording_files:
        playback_file = recording_files[0]
        print(f"Using recording file: {playback_file}")
    else:
        print("No recording files available")
        return
    
    # Set up camera manager
    camera_manager = camera_data_manager.CameraDataManager(playback_file=playback_file)
    
    try:
        camera_manager.connect()
        camera_manager.start_subscribing()
    except Exception as e:
        print(f"Error starting camera manager: {e}")
        return

    frame_count = 0
    last_frame_id = -1
    
    print("Starting playback analysis...")
    print("=" * 50)
    
    try:
        while camera_manager.is_running() and frame_count < 20:  # Limit to 20 frames for debugging
            fd = camera_manager.get_latest_frames()
            
            print(f"\nFrame {frame_count}:")
            print(f"  frame_count: {fd.frame_count}")
            print(f"  last_frame_id: {last_frame_id}")
            print(f"  camera_manager.is_running(): {camera_manager.is_running()}")
            
            if fd.frame_count == 0:
                print("  -> Frame count is 0, sleeping...")
                time.sleep(0.001)
                continue
                
            if fd.frame_count == last_frame_id:
                print("  -> Same frame ID as last time, sleeping...")
                time.sleep(0.001)
                continue
            
            print(f"  -> Processing new frame {fd.frame_count}")
            
            # Check frame data availability
            print(f"  RGB available: {fd.rgb is not None}")
            print(f"  Depth available: {fd.depth is not None}")
            print(f"  Motion available: {fd.motion is not None}")
            
            if fd.rgb is not None:
                try:
                    rgb_buf = fd.rgb.get_data()
                    print(f"  RGB data size: {len(rgb_buf)} bytes")
                    print(f"  RGB dimensions: {fd.rgb.width}x{fd.rgb.height}")
                    
                    # Try to decode
                    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                    if rgb_bgr is not None:
                        print(f"  RGB decoded successfully: {rgb_bgr.shape}")
                    else:
                        print("  RGB decode failed, trying raw reshape...")
                        w, h = fd.rgb.width, fd.rgb.height
                        rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))
                        print(f"  RGB raw reshape: {rgb_bgr.shape}")
                        
                except Exception as e:
                    print(f"  RGB processing error: {e}")
            
            if fd.depth is not None:
                try:
                    depth_img = fd.depth.get_data_2d().astype(np.float32)
                    print(f"  Depth data shape: {depth_img.shape}")
                    print(f"  Depth range: {depth_img.min():.3f} to {depth_img.max():.3f}")
                    non_zero = np.count_nonzero(depth_img)
                    print(f"  Non-zero depth pixels: {non_zero}/{depth_img.size} ({100*non_zero/depth_img.size:.1f}%)")
                except Exception as e:
                    print(f"  Depth processing error: {e}")
            
            last_frame_id = fd.frame_count
            frame_count += 1
            
            # Sleep a bit to see the output
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\nDebug interrupted...")
        
    finally:
        camera_manager.stop()


def create_test_recording():
    """Create a minimal test recording for debugging."""
    import gzip
    import pickle
    import os
    
    os.makedirs("recordings", exist_ok=True)
    
    # Create some mock frame data
    frames = []
    for i in range(5):
        frame_record = {
            'timestamp': i * 0.033,  # 30 FPS
            'frame_count': i + 1,
            'rgb': {
                'data': b'\x00\x01\x02' * (640 * 480),  # Mock RGB data
                'timestamp': 1234567890.0 + i * 0.033,
                'width': 640,
                'height': 480
            },
            'depth': {
                'raw_data': [100 + i] * (640 * 480),  # Mock depth data
                'timestamp': 1234567890.0 + i * 0.033,
                'width': 640,
                'height': 480
            },
            'motion': {
                'gyro': [0.1 * i, 0.2 * i, 0.3 * i],
                'accel': [9.8, 0.1 * i, 0.2 * i],
                'timestamp': 1234567890.0 + i * 0.033
            }
        }
        frames.append(frame_record)
    
    # Save to file
    filename = "recordings/debug_test.pkl.gz"
    with gzip.open(filename, 'wb') as f:
        for frame in frames:
            pickle.dump(frame, f)
    
    print(f"Created test recording: {filename}")


if __name__ == "__main__":
    main() 