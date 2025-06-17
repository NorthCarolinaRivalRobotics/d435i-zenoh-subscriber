#!/usr/bin/env python3
"""
Rerun visualization utilities for RGB-D data and feature matching.
"""
from __future__ import annotations

import numpy as np
import rerun as rr
from profiling import profiler


class RerunVisualizer:
    """Helper class for logging RGB-D data and feature matching to Rerun."""
    
    def __init__(self, app_name: str = "zenoh_d435i_live", spawn: bool = True):
        """Initialize Rerun visualizer."""
        rr.init(app_name, spawn=spawn)
    
    def log_rgb_image(self, rgb_img: np.ndarray, path: str = "camera/rgb") -> None:
        """Log RGB image to Rerun."""
        with profiler.timer("rerun_rgb_log"):
            rr.log(path, rr.Image(rgb_img))
    
    def log_depth_image(self, depth_img: np.ndarray, path: str = "camera/depth", 
                       meter: float = 1.0) -> None:
        """Log depth image to Rerun."""
        with profiler.timer("rerun_depth_log"):
            rr.log(path, rr.DepthImage(depth_img, meter=meter))
    
    def log_3d_matches(self, points_prev: np.ndarray, points_curr: np.ndarray,
                      path_prev: str = "matches/frame_prev",
                      path_curr: str = "matches/frame_curr",
                      path_lines: str = "matches/lines") -> None:
        """Log 3D point matches and connection lines to Rerun."""
        if len(points_prev) == 0 or len(points_curr) == 0:
            return
        
        with profiler.timer("rerun_matches_log"):
            # Log 3D points
            rr.log(path_prev, rr.Points3D(points_prev, colors=[255, 0, 0]))  # Red
            rr.log(path_curr, rr.Points3D(points_curr, colors=[0, 255, 0]))  # Green
            
            # Log connection lines
            line_strips = [[p, q] for p, q in zip(points_prev, points_curr)]
            rr.log(path_lines, rr.LineStrips3D(line_strips))
    
    def log_camera_pose(self, transform_matrix: np.ndarray, frame_id: int,
                       path_prefix: str = "frames") -> None:
        """Log camera pose as SE(3) transform to Rerun."""
        if transform_matrix is None:
            return
        
        with profiler.timer("rerun_pose_log"):
            path = f"{path_prefix}/{frame_id}/pose"
            rr.log(path, rr.Transform3D(
                mat3x3=transform_matrix[:3, :3],
                translation=transform_matrix[:3, 3]
            ))
    
    def log_frame_data(self, rgb_img: np.ndarray, depth_img: np.ndarray,
                      points_prev: np.ndarray | None = None,
                      points_curr: np.ndarray | None = None,
                      transform: np.ndarray | None = None,
                      frame_id: int | None = None) -> None:
        """Log complete frame data (convenience method)."""
        # Log images
        self.log_rgb_image(rgb_img)
        self.log_depth_image(depth_img)
        
        # Log matches if available
        if points_prev is not None and points_curr is not None:
            self.log_3d_matches(points_prev, points_curr)
        
        # Log camera pose if available
        if transform is not None and frame_id is not None:
            self.log_camera_pose(transform, frame_id) 