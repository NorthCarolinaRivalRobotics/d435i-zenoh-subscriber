#!/usr/bin/env python3
"""
Rerun visualization utilities for RGB-D data and feature matching.
"""
from __future__ import annotations

import numpy as np
import rerun as rr
from profiling import profiler
import cv2


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    # Use OpenCV's Rodrigues to convert rotation matrix to axis-angle
    axis_angle, _ = cv2.Rodrigues(R)
    return axis_angle.flatten()


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
    
    def log_pinhole_camera(self, K: np.ndarray, width: int, height: int, 
                          transform_matrix: np.ndarray, frame_id: int,
                          path_prefix: str = "camera_poses") -> None:
        """Log pinhole camera with pose using axis-angle representation."""
        if transform_matrix is None:
            return
            
        with profiler.timer("rerun_pinhole_log"):
            # Extract camera path
            camera_path = f"{path_prefix}/{frame_id}"
            
            # Log pinhole camera intrinsics
            rr.log(camera_path, rr.Pinhole(
                image_from_camera=K,
                width=width,
                height=height
            ))
            
            # Convert rotation matrix to axis-angle for the transform
            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]
            
            # Convert rotation to axis-angle representation
            axis_angle = rotation_matrix_to_axis_angle(R)
            
            # Log transform with axis-angle representation
            rr.log(camera_path, rr.Transform3D(
                rotation=rr.RotationAxisAngle(axis=axis_angle[:3], angle=np.linalg.norm(axis_angle)),
                translation=t
            ))
    
    def log_accumulated_camera_pose(self, T_accumulated: np.ndarray, K: np.ndarray, 
                                   width: int, height: int, frame_id: int) -> None:
        """Log the accumulated camera pose as a pinhole camera."""
        if T_accumulated is None:
            return
            
        with profiler.timer("rerun_accumulated_pose_log"):
            # Log only the current camera position as a pinhole camera
            camera_path = "trajectory/current_camera"
            
            # Log pinhole camera intrinsics at current position
            rr.log(camera_path, rr.Pinhole(
                image_from_camera=K,
                width=width,
                height=height
            ))
            
            # Convert rotation matrix to axis-angle for the transform
            R = T_accumulated[:3, :3]
            t = T_accumulated[:3, 3]
            
            # Convert rotation to axis-angle representation
            axis_angle = rotation_matrix_to_axis_angle(R)
            
            # Log transform with axis-angle representation
            rr.log(camera_path, rr.Transform3D(
                rotation=rr.RotationAxisAngle(axis=axis_angle[:3], angle=np.linalg.norm(axis_angle)),
                translation=t
            ))
    
    def log_camera_trajectory(self, positions: list[np.ndarray], 
                             path: str = "trajectory/path") -> None:
        """Log camera trajectory as a connected line and points."""
        if len(positions) < 1:
            return
            
        with profiler.timer("rerun_trajectory_log"):
            # Convert list of positions to numpy array
            positions_array = np.array(positions)
            
            # Log trajectory points
            rr.log(f"{path}/points", rr.Points3D(
                positions_array, 
                colors=[100, 150, 255],  # Light blue
                radii=0.02  # Small radius for trail points
            ))
            
            # Log trajectory as connected lines if we have multiple points
            if len(positions) > 1:
                # Create line segments connecting consecutive positions
                line_strips = [positions_array]  # Single connected line
                rr.log(f"{path}/lines", rr.LineStrips3D(
                    line_strips,
                    colors=[50, 100, 200]  # Darker blue for lines
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
            
    # === Additional fusion-specific visualization methods ===
    
    def log_coordinate_axes(self, origin: np.ndarray = np.zeros(3), 
                           rotation: np.ndarray = np.eye(3),
                           scale: float = 1.0,
                           path: str = "axes",
                           labels: list[str] = ["X", "Y", "Z"]) -> None:
        """Log coordinate axes at given origin with rotation."""
        with profiler.timer("rerun_axes_log"):
            # Create axis vectors
            axes = scale * np.eye(3)
            
            # Rotate axes
            rotated_axes = rotation @ axes
            
            # Create arrows for each axis
            origins = [origin, origin, origin]
            vectors = [rotated_axes[:, i] for i in range(3)]
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
            
            rr.log(path, rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                colors=colors,
                labels=labels
            ))
    
    def log_comparison_trajectory(self, positions1: list[np.ndarray], 
                                 positions2: list[np.ndarray],
                                 labels: tuple[str, str] = ("Reference", "Estimated"),
                                 colors: tuple[list[int], list[int]] = ([0, 255, 0], [255, 0, 0]),
                                 path_prefix: str = "comparison") -> None:
        """Log two trajectories for comparison (e.g., ground truth vs estimated)."""
        with profiler.timer("rerun_comparison_log"):
            # First trajectory
            if len(positions1) > 0:
                pos_array1 = np.array(positions1)
                rr.log(f"{path_prefix}/{labels[0]}/points", rr.Points3D(
                    pos_array1,
                    colors=colors[0],
                    radii=0.02
                ))
                
                if len(positions1) > 1:
                    rr.log(f"{path_prefix}/{labels[0]}/lines", rr.LineStrips3D(
                        [pos_array1],
                        colors=colors[0]
                    ))
            
            # Second trajectory
            if len(positions2) > 0:
                pos_array2 = np.array(positions2)
                rr.log(f"{path_prefix}/{labels[1]}/points", rr.Points3D(
                    pos_array2,
                    colors=colors[1],
                    radii=0.02
                ))
                
                if len(positions2) > 1:
                    rr.log(f"{path_prefix}/{labels[1]}/lines", rr.LineStrips3D(
                        [pos_array2],
                        colors=colors[1]
                    ))
                    
    def log_uncertainty_ellipsoid(self, position: np.ndarray, covariance: np.ndarray,
                                 n_std: float = 2.0, n_points: int = 50,
                                 path: str = "uncertainty/ellipsoid",
                                 color: list[int] = [255, 255, 0, 128]) -> None:
        """Log uncertainty ellipsoid from position and covariance matrix."""
        with profiler.timer("rerun_uncertainty_log"):
            # Extract position covariance (3x3) from full covariance if needed
            if covariance.shape == (6, 6):
                pos_cov = covariance[3:6, 3:6]  # Translation part
            elif covariance.shape == (3, 3):
                pos_cov = covariance
            else:
                return
                
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
            
            # Generate points on unit sphere
            u = np.linspace(0, 2 * np.pi, n_points)
            v = np.linspace(0, np.pi, n_points // 2)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Scale by eigenvalues (n_std standard deviations)
            scale = n_std * np.sqrt(eigenvalues)
            points = np.stack([x.ravel() * scale[0], 
                              y.ravel() * scale[1], 
                              z.ravel() * scale[2]], axis=1)
            
            # Rotate by eigenvectors and translate
            points = (eigenvectors @ points.T).T + position
            
            # Log as point cloud
            rr.log(path, rr.Points3D(
                points,
                colors=color,
                radii=0.005
            ))
            
    def log_sensor_status(self, imu_rate: float, vo_rate: float, fusion_rate: float,
                         imu_latency: float = 0.0, vo_latency: float = 0.0,
                         path_prefix: str = "status") -> None:
        """Log sensor status information."""
        with profiler.timer("rerun_status_log"):
            # Log rates
            rr.log(f"{path_prefix}/rates/imu", rr.Scalar(imu_rate))
            rr.log(f"{path_prefix}/rates/visual_odometry", rr.Scalar(vo_rate))
            rr.log(f"{path_prefix}/rates/fusion", rr.Scalar(fusion_rate))
            
            # Log latencies if available
            if imu_latency > 0:
                rr.log(f"{path_prefix}/latency/imu", rr.Scalar(imu_latency * 1000))  # ms
            if vo_latency > 0:
                rr.log(f"{path_prefix}/latency/visual_odometry", rr.Scalar(vo_latency * 1000))  # ms 