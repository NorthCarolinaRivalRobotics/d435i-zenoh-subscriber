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
from vision_utils import estimate_frame_transform
from visualization import RerunVisualizer


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

    try:
        while sub.is_running():
            # Get latest frame data
            fd = sub.get_latest_frames()
            if fd.frame_count == 0:
                time.sleep(0.05)
                continue
            if last_frame is not None and fd.frame_count == last_frame["id"]:
                time.sleep(0.02)
                continue

            # Process frame data
            rgb_buf = fd.rgb.get_data()
            w, h = fd.rgb.width, fd.rgb.height

            # Decode RGB image
            rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))

            rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth_img = fd.depth.get_data_2d().astype(np.float32)

            # Log basic frame data
            visualizer.log_rgb_image(rgb_img)
            visualizer.log_depth_image(depth_img, meter=1.0)

            # Perform feature matching if we have a previous frame
            if last_frame is not None:
                P1, P2, T = estimate_frame_transform(
                    last_frame["rgb"], last_frame["depth"], 
                    rgb_img, depth_img,
                    camera_cal.K_rgb, camera_cal.K_depth, 
                    camera_cal.T_rgb_to_depth, 
                    depth_scale=1.0
                )

                # Log feature matches
                if len(P1) > 0:
                    visualizer.log_3d_matches(P1, P2)

                # Log camera pose
                if T is not None:
                    visualizer.log_camera_pose(T, frame_id)

            # Update state for next iteration
            last_frame = {"rgb": rgb_img, "depth": depth_img, "id": fd.frame_count}
            frame_id += 1

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        sub.stop()


if __name__ == "__main__":
    main()
