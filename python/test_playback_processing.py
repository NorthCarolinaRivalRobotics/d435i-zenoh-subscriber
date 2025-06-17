#!/usr/bin/env python3
"""
Debug playback processing to see what's happening with pose calculation.
"""
import sys
import time
import cv2
import numpy as np
sys.path.insert(0, '.')

from data_source import PlaybackCameraDataSource

def main():
    """Debug playback processing."""
    recording_file = "../recordings/camera_data_20250616_234956.pkl.gz"
    
    print(f"Testing playback with: {recording_file}")
    
    # Create playback data source
    playback_source = PlaybackCameraDataSource(
        recording_path=recording_file,
        loop=False,
        realtime=False
    )
    
    try:
        playback_source.connect()
        playback_source.start_subscribing()
        
        frame_count = 0
        last_frame_id = -1
        successful_frames = 0
        failed_frames = 0
        
        print("Starting frame processing debug...")
        
        for i in range(20):  # Test first 20 frames
            fd = playback_source.get_latest_frames()
            
            if fd.frame_count == 0:
                print(f"Frame {i}: No data (frame_count=0)")
                time.sleep(0.001)
                continue
                
            if fd.frame_count == last_frame_id:
                print(f"Frame {i}: Same frame ID {fd.frame_count} (duplicate)")
                time.sleep(0.001)
                continue
                
            # New frame!
            frame_count += 1
            last_frame_id = fd.frame_count
            
            print(f"\\nFrame {frame_count} (ID: {fd.frame_count}):")
            
            # Check RGB data
            try:
                if fd.rgb is not None:
                    rgb_buf = fd.rgb.get_data()
                    print(f"  RGB: {fd.rgb.width}x{fd.rgb.height}, {len(rgb_buf)} bytes")
                    
                    # Try to decode
                    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                    if rgb_bgr is not None:
                        print(f"  RGB decode: SUCCESS {rgb_bgr.shape}")
                        successful_frames += 1
                    else:
                        print(f"  RGB decode: FAILED")
                        failed_frames += 1
                else:
                    print(f"  RGB: None")
                    failed_frames += 1
                    
            except Exception as e:
                print(f"  RGB error: {e}")
                failed_frames += 1
                
            # Check depth data  
            try:
                if fd.depth is not None:
                    depth_img = fd.depth.get_data_2d()
                    print(f"  Depth: {fd.depth.width}x{fd.depth.height}, shape {depth_img.shape}")
                    print(f"  Depth range: {depth_img.min():.3f} - {depth_img.max():.3f}")
                else:
                    print(f"  Depth: None")
            except Exception as e:
                print(f"  Depth error: {e}")
                
        print(f"\\nSummary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Successful RGB decodes: {successful_frames}")
        print(f"  Failed RGB decodes: {failed_frames}")
        if successful_frames + failed_frames > 0:
            print(f"  Success rate: {successful_frames/(successful_frames+failed_frames)*100:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        playback_source.stop()

if __name__ == "__main__":
    main() 