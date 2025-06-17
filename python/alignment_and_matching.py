#!/usr/bin/env python3
"""
Zenoh RealSense D435i subscriber with live Rerun visualisation.

* Uses *only* the unified rr.log(path, object) API (§ "Entity Path Hierarchy").
* Streams RGB, depth, 3-D matches, match-lines, and SE(3) frame pose.
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import zenoh_d435i_subscriber as zd435i

from camera_config import CameraCalibration
from vision_utils import CAMERA_TO_ROBOT_FRAME, estimate_frame_transform
from visualization import RerunVisualizer
from profiling import profiler


def main() -> None:
    """Main application loop for RGB-D feature tracking and visualization."""
    
    # Initialize components
    visualizer = RerunVisualizer("zenoh_d435i_live", spawn=True)
    camera_cal = CameraCalibration.create_default_d435i()
    
    # Setup Zenoh subscriber
    sub = zd435i.ZenohD435iSubscriber()
    sub.connect()
    sub.start_subscribing()

    last_frame = None
    frame_id = 0
    
    # Performance optimization parameters
    FEATURE_MATCHING_INTERVAL = 3  # Only do feature matching every N frames
    MAX_PROCESSING_TIME = 0.030    # Target 30ms max processing time per frame
    MIN_SLEEP_TIME = 0.001         # Minimum sleep time (1ms)
    
    # Statistics
    frame_times = []
    processing_times = []
    last_stats_time = time.time()

    # SE(3) matrix at 0 rotation and 0 translation
    T_Accumulated = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
    
    # Accumulate in camera coordinates first, then transform
    T_Accumulated_Camera = np.array([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])
    
    # Track camera trajectory positions
    camera_trajectory = []
    MAX_TRAJECTORY_LENGTH = 500  # Limit trajectory length for performance

    try:
        while sub.is_running():
            frame_start = time.perf_counter()
            
            # Get latest frame data with minimal sleep
            with profiler.timer("frame_acquisition"):
                fd = sub.get_latest_frames()
                if fd.frame_count == 0:
                    time.sleep(MIN_SLEEP_TIME)
                    continue
                if last_frame is not None and fd.frame_count == last_frame["id"]:
                    time.sleep(MIN_SLEEP_TIME)  # Reduced from 0.02
                    continue

            # Process frame data
            with profiler.timer("image_decoding"):
                try:
                    rgb_buf = fd.rgb.get_data()
                    w, h = fd.rgb.width, fd.rgb.height

                    # Decode RGB image
                    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                    if rgb_bgr is None:
                        rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))

                    rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    depth_img = fd.depth.get_data_2d().astype(np.float32)
                except Exception as e:
                    print(f"Error decoding images: {e}")
                    time.sleep(MIN_SLEEP_TIME)
                    continue

            # Log basic frame data (this should be fast)
            with profiler.timer("visualization_basic"):
                visualizer.log_rgb_image(rgb_img)
                visualizer.log_depth_image(depth_img, meter=1.0)

            # Perform feature matching only occasionally to reduce latency
            do_feature_matching = (
                last_frame is not None and 
                frame_id % FEATURE_MATCHING_INTERVAL == 0
            )
            
            if do_feature_matching:
                with profiler.timer("feature_matching_full"):
                    P1, P2, T = estimate_frame_transform(
                        last_frame["rgb"], last_frame["depth"], 
                        rgb_img, depth_img,
                        camera_cal.K_rgb, camera_cal.K_depth, 
                        camera_cal.T_rgb_to_depth, 
                        depth_scale=1.0  # Depth data is already in meters from decode_u16_to_meters()
                    )

                    # print(f"Transform: {T}")

                    # Log feature matches
                    if len(P1) > 0:
                        with profiler.timer("visualization_matches"):
                            visualizer.log_3d_matches(P1, P2)

                    # Log camera pose
                    if T is not None:
                        with profiler.timer("visualization_pose"):
                            # Log the relative transform using the original method
                            visualizer.log_camera_pose(T, frame_id)
                            
                            # Debug: Print individual transform components
                            print(f"Raw pose T translation: {T[:3, 3]}")
                            print(f"Raw pose T scale (det): {np.linalg.det(T[:3, :3]):.6f}")
                            
                            # Apply coordinate transform
                            T_robot = CAMERA_TO_ROBOT_FRAME @ T
                            print(f"Robot frame translation: {T_robot[:3, 3]}")
                            print(f"Robot frame scale (det): {np.linalg.det(T_robot[:3, :3]):.6f}")
                            
                            # Accumulate
                            T_Accumulated_Camera = T_Accumulated_Camera @ T
                            print(f"Accumulated Transform in Camera Coordinates:\n {T_Accumulated_Camera}")
                            print(f"Accumulated position: {T_Accumulated_Camera[:3, 3]}")
                            print(f"Position breakdown - X(fwd): {T_Accumulated_Camera[0,3]:.3f}, Y(left): {T_Accumulated_Camera[1,3]:.3f}, Z(up): {T_Accumulated_Camera[2,3]:.3f}")
                            print(f"Total distance moved: {np.linalg.norm(T_Accumulated_Camera[:3, 3]):.3f}m")
                            print("---")

                            # Transform to robot frame
                            T_Accumulated = CAMERA_TO_ROBOT_FRAME @ T_Accumulated_Camera
                            print(f"Accumulated Transform in Robot Frame:\n {T_Accumulated}")
                            print(f"Accumulated position: {T_Accumulated[:3, 3]}")
                            print(f"Position breakdown - X(fwd): {T_Accumulated[0,3]:.3f}, Y(left): {T_Accumulated[1,3]:.3f}, Z(up): {T_Accumulated[2,3]:.3f}")
                            print(f"Total distance moved: {np.linalg.norm(T_Accumulated[:3, 3]):.3f}m")
                            print("---")
                            
                            # Add current position to trajectory
                            current_position = T_Accumulated_Camera[:3, 3]
                            camera_trajectory.append(current_position.copy())
                            
                            # Limit trajectory length for performance
                            if len(camera_trajectory) > MAX_TRAJECTORY_LENGTH:
                                camera_trajectory = camera_trajectory[-MAX_TRAJECTORY_LENGTH:]
                            
                            # Log the current camera pose as a single pinhole camera
                            visualizer.log_accumulated_camera_pose(
                                T_accumulated=T_Accumulated_Camera,
                                K=camera_cal.K_rgb,
                                width=w,
                                height=h,
                                frame_id=frame_id
                            )
                            
                            # Log the trajectory trail
                            if len(camera_trajectory) > 1:
                                visualizer.log_camera_trajectory(camera_trajectory)

            # Update state for next iteration
            last_frame = {"rgb": rgb_img, "depth": depth_img, "id": fd.frame_count}
            frame_id += 1
            
            # Track performance
            frame_time = time.perf_counter() - frame_start
            frame_times.append(frame_time)
            processing_times.append(frame_time)
            
            # Print performance stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                if frame_times:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    max_frame_time = max(frame_times)
                    print(f"Performance: {fps:.1f} FPS (avg: {avg_frame_time*1000:.1f}ms, max: {max_frame_time*1000:.1f}ms)")
                    
                    # Reset for next period
                    frame_times = []
                    last_stats_time = current_time
            
            # Adaptive sleep to prevent CPU spinning while maintaining responsiveness
            if frame_time < MAX_PROCESSING_TIME:
                sleep_time = min(MIN_SLEEP_TIME, (MAX_PROCESSING_TIME - frame_time) * 0.1)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nShutting down...")
        
        # Print final profiling results
        print("\nFinal Performance Analysis:")
        profiler.print_results(min_calls=1)
        
    finally:
        sub.stop()


if __name__ == "__main__":
    main()
