#!/usr/bin/env python3
"""
Minimal latency test - just RGB/depth visualization, no feature matching.
This will help isolate if the latency is in Rust decompression or Python processing.

Now supports recording and playback:
- Live mode (default): python alignment_minimal.py
- Recording mode: python alignment_minimal.py --record
- Playback mode: python alignment_minimal.py --playback recordings/camera_data_20231201_120000.pkl.gz
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
from visualization import RerunVisualizer
from profiling import profiler


def main() -> None:
    """Minimal loop to test latency without feature matching."""
    
    # Set up camera manager with command line arguments
    camera_manager, args = setup_camera_manager("Minimal latency test with recording/playback support")
    
    print("No feature matching - just RGB/depth visualization")
    
    # Initialize components
    mode_str = camera_manager.get_mode()
    visualizer = RerunVisualizer(f"minimal_test_{mode_str}", spawn=True)
    
    # Connect and start
    try:
        camera_manager.connect()
        camera_manager.start_subscribing()
    except Exception as e:
        print(f"Error starting camera manager: {e}")
        return

    frame_count = 0
    last_frame_id = -1
    
    # Performance tracking
    frame_times = []
    rust_times = []
    decode_times = []
    vis_times = []
    last_stats_time = time.time()

    try:
        while camera_manager.is_running():
            loop_start = time.perf_counter()
            
            # Measure frame acquisition time
            rust_start = time.perf_counter()
            fd = camera_manager.get_latest_frames()
            rust_time = time.perf_counter() - rust_start
            rust_times.append(rust_time)
            
            if fd.frame_count == 0:
                time.sleep(0.0005)  # 0.5ms
                continue
            if fd.frame_count == last_frame_id:
                time.sleep(0.0005)
                continue
                
            last_frame_id = fd.frame_count

            # Measure decode time
            decode_start = time.perf_counter()
            
            with profiler.timer("get_rgb_data"):
                rgb_buf = fd.rgb.get_data()
            
            with profiler.timer("decode_jpeg"):
                rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                if rgb_bgr is None:
                    w, h = fd.rgb.width, fd.rgb.height
                    rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))
            
            with profiler.timer("color_convert"):
                rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            
            with profiler.timer("get_depth_data"):
                depth_img = fd.depth.get_data_2d().astype(np.float32)
            
            decode_time = time.perf_counter() - decode_start
            decode_times.append(decode_time)

            # Measure visualization time
            vis_start = time.perf_counter()
            
            with profiler.timer("rerun_rgb"):
                visualizer.log_rgb_image(rgb_img)
            
            with profiler.timer("rerun_depth"):
                visualizer.log_depth_image(depth_img, meter=1.0)
            
            vis_time = time.perf_counter() - vis_start
            vis_times.append(vis_time)

            frame_count += 1
            
            # Total frame time
            total_time = time.perf_counter() - loop_start
            frame_times.append(total_time)
            
            # Print detailed stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                if frame_times:
                    avg_total = sum(frame_times) / len(frame_times) * 1000
                    avg_rust = sum(rust_times) / len(rust_times) * 1000
                    avg_decode = sum(decode_times) / len(decode_times) * 1000
                    avg_vis = sum(vis_times) / len(vis_times) * 1000
                    fps = len(frame_times) / 5.0
                    
                    mode_prefix = camera_manager.get_status_string()
                    print(f"{mode_prefix} FPS: {fps:.1f} | Total: {avg_total:.1f}ms | "
                          f"Data: {avg_rust:.1f}ms | Decode: {avg_decode:.1f}ms | "
                          f"Vis: {avg_vis:.1f}ms")
                    
                    # Show percentages
                    print(f"  Breakdown: Data {(avg_rust/avg_total)*100:.1f}%, "
                          f"Decode {(avg_decode/avg_total)*100:.1f}%, "
                          f"Vis {(avg_vis/avg_total)*100:.1f}%")
                    
                    # Reset for next period
                    frame_times = []
                    rust_times = []
                    decode_times = []
                    vis_times = []
                    last_stats_time = current_time

    except KeyboardInterrupt:
        print(f"\nShutting down {mode_str} mode...")
        
        # Print final profiling results
        print("\nDetailed Performance Analysis:")
        profiler.print_results(min_calls=1)
        
    finally:
        camera_manager.stop()


if __name__ == "__main__":
    main() 