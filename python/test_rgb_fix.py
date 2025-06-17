#!/usr/bin/env python3
"""
Test the RGB fix for playback.
"""
import pickle
import gzip
import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
from frame_reconstruction import PlaybackFrameData

def test_rgb_fix(recording_file):
    """Test RGB data fix during playback."""
    print(f"Testing RGB fix with: {recording_file}")
    
    with gzip.open(recording_file, 'rb') as f:
        frame_count = 0
        try:
            while frame_count < 3:  # Test first 3 frames
                frame_record = pickle.load(f)
                frame_count += 1
                
                print(f"\nFrame {frame_count}:")
                print(f"  Frame count: {frame_record['frame_count']}")
                
                if frame_record['rgb'] is not None:
                    # Create playback frame data
                    playback_frame = PlaybackFrameData(frame_record)
                    rgb_frame = playback_frame.rgb
                    
                    print(f"  RGB dimensions: {rgb_frame.width}x{rgb_frame.height}")
                    
                    try:
                        # Get the JPEG data (this will trigger conversion)
                        jpeg_data = rgb_frame.get_data()
                        print(f"  JPEG data length: {len(jpeg_data)}")
                        print(f"  First 10 bytes: {jpeg_data[:10]}")
                        
                        # Try to decode with cv2
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            print(f"  ✅ Successfully decoded RGB image: {img.shape}")
                            print(f"  Image dtype: {img.dtype}")
                            print(f"  Image min/max: {img.min()}/{img.max()}")
                        else:
                            print(f"  ❌ Failed to decode RGB image")
                            
                    except Exception as e:
                        print(f"  ❌ Error processing RGB: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  No RGB data")
                    
        except EOFError:
            print(f"\nProcessed {frame_count} frames total")

if __name__ == "__main__":
    import glob
    
    # Look for recording files
    recording_files = glob.glob("../recordings/*.pkl.gz")
    if not recording_files:
        print("No recording files found")
        sys.exit(1)
    
    recording_file = recording_files[0]
    print(f"Using recording file: {recording_file}")
    test_rgb_fix(recording_file) 