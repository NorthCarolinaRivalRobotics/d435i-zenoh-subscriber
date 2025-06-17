#!/usr/bin/env python3
"""
Optimized version that reduces Rerun load to eliminate latency.
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import zenoh_d435i_subscriber as zd435i

from camera_config import CameraCalibration
from vision_utils import estimate_frame_transform
from visualization import RerunVisualizer
from profiling import profiler


def main() -> None:
    """Optimized loop with reduced Rerun load."""
    
    print("Starting OPTIMIZED version with reduced Rerun load...")
    
    # Initialize components
    visualizer = RerunVisualizer("optimized_low_latency", spawn=True)
    camera_cal = CameraCalibration.create_default_d435i()
    
    # Setup Zenoh subscriber
    sub = zd435i.ZenohD435iSubscriber()
    sub.connect()
    sub.start_subscribing()

    last_frame = None
    frame_id = 0
    
    # Optimization parameters
    RERUN_DECIMATION = 2  # Only log every 2nd frame to Rerun
    IMAGE_DOWNSCALE = 2   # Downscale images for Rerun
    FEATURE_INTERVAL = 4  # Feature matching every 4th frame
    
    # Performance tracking
    frame_times = []
    last_stats_time = time.time()

    try:
        while sub.is_running():
            loop_start = time.perf_counter()
            
            # Fast frame acquisition
            fd = sub.get_latest_frames()
            if fd.frame_count == 0:
                time.sleep(0.0005)
                continue
            if last_frame is not None and fd.frame_count == last_frame["id"]:
                time.sleep(0.0005)
                continue

            # Fast decode
            rgb_buf = fd.rgb.get_data()
            w, h = fd.rgb.width, fd.rgb.height

            rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))

            rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth_img = fd.depth.get_data_2d().astype(np.float32)

            # Only log to Rerun occasionally to reduce load
            if frame_id % RERUN_DECIMATION == 0:
                with profiler.timer("rerun_logging"):
                    # Downscale images for Rerun to reduce bandwidth
                    rgb_small = cv2.resize(rgb_img, (w//IMAGE_DOWNSCALE, h//IMAGE_DOWNSCALE))
                    depth_small = cv2.resize(depth_img, (w//IMAGE_DOWNSCALE, h//IMAGE_DOWNSCALE))
                    
                    visualizer.log_rgb_image(rgb_small)
                    visualizer.log_depth_image(depth_small, meter=1.0)

            # Feature matching less frequently
            if last_frame is not None and frame_id % FEATURE_INTERVAL == 0:
                with profiler.timer("feature_matching"):
                    P1, P2, T = estimate_frame_transform(
                        last_frame["rgb"], last_frame["depth"], 
                        rgb_img, depth_img,
                        camera_cal.K_rgb, camera_cal.K_depth, 
                        camera_cal.T_rgb_to_depth, 
                        depth_scale=1.0
                    )

                    # Only log feature matches to Rerun occasionally
                    if len(P1) > 0 and frame_id % RERUN_DECIMATION == 0:
                        with profiler.timer("rerun_features"):
                            visualizer.log_3d_matches(P1, P2)

                    if T is not None and frame_id % RERUN_DECIMATION == 0:
                        with profiler.timer("rerun_pose"):
                            visualizer.log_camera_pose(T, frame_id)

            # Update state
            last_frame = {"rgb": rgb_img, "depth": depth_img, "id": fd.frame_count}
            frame_id += 1
            
            # Performance tracking
            total_time = time.perf_counter() - loop_start
            frame_times.append(total_time)
            
            # Print stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                if frame_times:
                    avg_time = sum(frame_times) / len(frame_times) * 1000
                    fps = len(frame_times) / 5.0
                    max_time = max(frame_times) * 1000
                    
                    print(f"Processing: {fps:.1f} FPS | "
                          f"Avg: {avg_time:.1f}ms | Max: {max_time:.1f}ms | "
                          f"Frames: {frame_id}")
                    
                    # Reset for next period
                    frame_times = []
                    last_stats_time = current_time

    except KeyboardInterrupt:
        print("\nShutting down...")
        
        # Print final profiling results
        print("\nFinal Performance Analysis:")
        profiler.print_results(min_calls=1)
        
    finally:
        sub.stop()


if __name__ == "__main__":
    main() 