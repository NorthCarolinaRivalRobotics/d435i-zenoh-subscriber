#!/usr/bin/env python3
"""
Debug RGB data during playback.
"""
import pickle
import gzip
import cv2
import numpy as np

def debug_rgb_data(recording_file):
    """Debug RGB data in recording file."""
    print(f"Debugging RGB data in: {recording_file}")
    
    with gzip.open(recording_file, 'rb') as f:
        frame_count = 0
        try:
            while frame_count < 5:  # Check first 5 frames
                frame = pickle.load(f)
                frame_count += 1
                
                print(f"\nFrame {frame_count}:")
                print(f"  Frame count: {frame['frame_count']}")
                print(f"  Timestamp: {frame['timestamp']}")
                
                if frame['rgb'] is not None:
                    rgb_data = frame['rgb']['data']
                    print(f"  RGB data type: {type(rgb_data)}")
                    print(f"  RGB data length: {len(rgb_data) if rgb_data else 0}")
                    print(f"  RGB dimensions: {frame['rgb']['width']}x{frame['rgb']['height']}")
                    
                    if rgb_data and len(rgb_data) > 0:
                        # Try to decode as JPEG
                        try:
                            print(f"  First 20 bytes: {rgb_data[:20]}")
                            
                            # Convert to numpy array for cv2.imdecode
                            nparr = np.frombuffer(rgb_data, np.uint8)
                            print(f"  Numpy array shape: {nparr.shape}")
                            
                            # Try to decode
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is not None:
                                print(f"  ✅ Successfully decoded RGB image: {img.shape}")
                            else:
                                print(f"  ❌ Failed to decode RGB image")
                                
                                # Check if it starts with JPEG magic bytes
                                if rgb_data[:2] == b'\\xff\\xd8':
                                    print(f"  Has JPEG magic bytes")
                                else:
                                    print(f"  Missing JPEG magic bytes: {rgb_data[:2]}")
                                    
                        except Exception as e:
                            print(f"  ❌ Error decoding RGB: {e}")
                else:
                    print(f"  No RGB data")
                    
        except EOFError:
            print(f"\\nProcessed {frame_count} frames total")

if __name__ == "__main__":
    import sys
    import os
    
    # Check for recording files
    if len(sys.argv) > 1:
        recording_file = sys.argv[1]
    else:
        # Look for recording files
        import glob
        recording_files = glob.glob("../recordings/*.pkl.gz")
        if not recording_files:
            print("No recording files found")
            sys.exit(1)
        recording_file = recording_files[0]
        print(f"Using recording file: {recording_file}")
    
    debug_rgb_data(recording_file) 