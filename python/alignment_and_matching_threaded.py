#!/usr/bin/env python3
"""
Optimized Zenoh RealSense D435i subscriber with threaded processing.

This version separates visualization from feature matching to minimize latency:
- Main thread: Fast RGB/depth visualization only
- Background thread: Feature matching and pose estimation

This should significantly reduce the 0.5s latency you're experiencing.
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import threading
import queue
from dataclasses import dataclass
from typing import Optional
import zenoh_d435i_subscriber as zd435i

from camera_config import CameraCalibration
from vision_utils import estimate_frame_transform
from visualization import RerunVisualizer
from profiling import profiler


@dataclass
class FrameData:
    """Container for frame data to pass between threads."""
    rgb_img: np.ndarray
    depth_img: np.ndarray
    frame_id: int
    timestamp: float


class FeatureMatchingWorker:
    """Background worker for feature matching and pose estimation."""
    
    def __init__(self, camera_cal: CameraCalibration, visualizer: RerunVisualizer):
        self.camera_cal = camera_cal
        self.visualizer = visualizer
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to reduce latency
        self.running = False
        self.thread = None
        self.last_frame = None
        
    def start(self):
        """Start the background processing thread."""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the background processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def submit_frame(self, frame_data: FrameData):
        """Submit a frame for processing (non-blocking)."""
        try:
            # Drop old frames if queue is full to maintain low latency
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            # Queue is full, drop the frame to avoid latency buildup
            try:
                self.frame_queue.get_nowait()  # Remove old frame
                self.frame_queue.put_nowait(frame_data)  # Add new frame
            except queue.Empty:
                pass
                
    def _process_loop(self):
        """Main processing loop for the background thread."""
        while self.running:
            try:
                # Get frame with timeout to allow checking running flag
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if self.last_frame is not None:
                    with profiler.timer("feature_matching_background"):
                        P1, P2, T = estimate_frame_transform(
                            self.last_frame.rgb_img, self.last_frame.depth_img,
                            frame_data.rgb_img, frame_data.depth_img,
                            self.camera_cal.K_rgb, self.camera_cal.K_depth,
                            self.camera_cal.T_rgb_to_depth,
                            depth_scale=1.0
                        )
                        
                        # Log results in background thread
                        if len(P1) > 0:
                            with profiler.timer("visualization_matches_bg"):
                                self.visualizer.log_3d_matches(P1, P2)
                                
                        if T is not None:
                            with profiler.timer("visualization_pose_bg"):
                                self.visualizer.log_camera_pose(T, frame_data.frame_id)
                
                self.last_frame = frame_data
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in feature matching worker: {e}")


def main() -> None:
    """Main application loop optimized for minimal latency."""
    
    # Initialize components
    visualizer = RerunVisualizer("zenoh_d435i_live", spawn=True)
    camera_cal = CameraCalibration.create_default_d435i()
    
    # Setup Zenoh subscriber
    sub = zd435i.ZenohD435iSubscriber()
    sub.connect()
    sub.start_subscribing()
    
    # Start background feature matching worker
    feature_worker = FeatureMatchingWorker(camera_cal, visualizer)
    feature_worker.start()

    frame_id = 0
    last_frame_count = -1
    
    # Performance tracking
    frame_times = []
    visualization_times = []
    last_stats_time = time.time()
    
    # Minimal sleep time to reduce latency
    MIN_SLEEP_TIME = 0.0005  # 0.5ms
    
    print("Starting optimized visualization loop...")
    print("Feature matching will run in background thread")

    try:
        while sub.is_running():
            loop_start = time.perf_counter()
            
            # Get latest frame data with minimal blocking
            with profiler.timer("frame_acquisition_fast"):
                fd = sub.get_latest_frames()
                
                # Quick checks with minimal sleep
                if fd.frame_count == 0:
                    time.sleep(MIN_SLEEP_TIME)
                    continue
                if fd.frame_count == last_frame_count:
                    time.sleep(MIN_SLEEP_TIME)
                    continue
                    
                last_frame_count = fd.frame_count

            # Fast image processing - minimize time here
            vis_start = time.perf_counter()
            
            with profiler.timer("image_decoding_fast"):
                rgb_buf = fd.rgb.get_data()
                w, h = fd.rgb.width, fd.rgb.height

                # Fast decode - avoid extra copies when possible
                rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                if rgb_bgr is None:
                    rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))

                rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                depth_img = fd.depth.get_data_2d().astype(np.float32)

            # Log to Rerun immediately for minimal latency
            with profiler.timer("visualization_immediate"):
                visualizer.log_rgb_image(rgb_img)
                visualizer.log_depth_image(depth_img, meter=1.0)
            
            vis_time = time.perf_counter() - vis_start
            visualization_times.append(vis_time)
            
            # Submit frame to background worker for feature matching
            # This should be very fast (just putting in queue)
            frame_data = FrameData(
                rgb_img=rgb_img.copy(),  # Copy to avoid sharing data with background thread
                depth_img=depth_img.copy(),
                frame_id=frame_id,
                timestamp=time.time()
            )
            feature_worker.submit_frame(frame_data)
            
            frame_id += 1
            
            # Performance tracking
            total_frame_time = time.perf_counter() - loop_start
            frame_times.append(total_frame_time)
            
            # Print stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                if frame_times:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    avg_vis_time = sum(visualization_times) / len(visualization_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    max_frame_time = max(frame_times)
                    
                    print(f"Main Loop: {fps:.1f} FPS | "
                          f"Avg: {avg_frame_time*1000:.1f}ms | "
                          f"Max: {max_frame_time*1000:.1f}ms | "
                          f"Vis: {avg_vis_time*1000:.1f}ms")
                    
                    # Reset for next period
                    frame_times = []
                    visualization_times = []
                    last_stats_time = current_time
            
            # Minimal sleep only if we're ahead of target framerate
            if total_frame_time < 0.010:  # If less than 10ms (100+ FPS)
                time.sleep(MIN_SLEEP_TIME)

    except KeyboardInterrupt:
        print("\nShutting down...")
        
        # Print detailed profiling results
        print("\nPerformance Analysis:")
        profiler.print_results(min_calls=1)
        
    finally:
        feature_worker.stop()
        sub.stop()


if __name__ == "__main__":
    main() 