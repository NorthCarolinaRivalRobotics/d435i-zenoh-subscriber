#!/usr/bin/env python3
"""
Computer vision and geometry utilities for RGB-D processing.
"""
from __future__ import annotations

import cv2
import numpy as np
from profiling import profiler


def rgb_pixel_to_xyz(depth_reg: np.ndarray,
                     K_rgb: np.ndarray,
                     pixel: tuple[int, int],
                     depth_scale: float) -> np.ndarray | None:
    """Back-project one RGB pixel to 3-D in the RGB camera frame."""
    u, v = pixel
    z = depth_reg[v, u] / depth_scale
    if z == 0:
        return None
    fx, fy = K_rgb[0, 0], K_rgb[1, 1]
    cx, cy = K_rgb[0, 2], K_rgb[1, 2]
    return np.array([(u - cx) * z / fx,
                     (v - cy) * z / fy,
                     z], np.float32)


def estimate_frame_transform(rgb1: np.ndarray, depth1: np.ndarray,
                             rgb2: np.ndarray, depth2: np.ndarray,
                             K_rgb: np.ndarray, K_depth: np.ndarray,
                             T_rgb_depth: np.ndarray,
                             depth_scale: float = 1000.0,
                             n_keypoints: int = 800,
                             max_matches: int = 300):
    """
    ORB-match two RGB-D frames and return 3-D correspondences + SE(3) pose.
    
    Args:
        rgb1, rgb2: RGB images
        depth1, depth2: Depth images
        K_rgb: RGB camera intrinsic matrix
        K_depth: Depth camera intrinsic matrix
        T_rgb_depth: Transform from depth to RGB camera
        depth_scale: Scale factor for depth values
        n_keypoints: Number of ORB keypoints to detect
        max_matches: Maximum number of matches to use
        
    Returns:
        tuple: (points_3d_frame1, points_3d_frame2, transform_matrix)
    """
    H, W = rgb1.shape[:2]

    # Depth→RGB registration
    with profiler.timer("depth_registration"):
        reg1 = cv2.rgbd.registerDepth(
            K_depth.astype(np.float32),
            K_rgb.astype(np.float32),
            None,                                   # distortion of RGB cam
            T_rgb_depth.astype(np.float32),
            depth1.astype(np.float32),
            (int(W), int(H)),                       # make sure these are int
            depthDilation=True,
        )
        reg2 = cv2.rgbd.registerDepth(
            K_depth.astype(np.float32),
            K_rgb.astype(np.float32),
            None,                                   # distortion of RGB cam (same as reg1)
            T_rgb_depth.astype(np.float32),
            depth2.astype(np.float32),
            (int(W), int(H)),                       # make sure these are int
            depthDilation=True
        )

    # ORB feature detection
    with profiler.timer("orb_detection"):
        orb = cv2.ORB_create(n_keypoints)
        kp1, des1 = orb.detectAndCompute(rgb1, None)
        kp2, des2 = orb.detectAndCompute(rgb2, None)
    
    # Check if descriptors were found
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return np.array([]), np.array([]), None
    
    # Feature matching
    with profiler.timer("feature_matching"):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            # Ensure matches is a list and not empty
            if not isinstance(matches, list):
                matches = list(matches) if matches else []
            if len(matches) == 0:
                return np.array([]), np.array([]), None
                
            matches.sort(key=lambda m: m.distance)
            matches = matches[:max_matches]
        except Exception as e:
            print(f"Error in matching: {e}")
            return np.array([]), np.array([]), None

    # Build 3-D ↔ 2-D correspondences
    with profiler.timer("correspondence_building"):
        obj_pts, img_pts, p2_pts = [], [], []
        for m in matches:
            p_xyz = rgb_pixel_to_xyz(reg1, K_rgb,
                                     tuple(map(int, kp1[m.queryIdx].pt)),
                                     depth_scale)
            q_xyz = rgb_pixel_to_xyz(reg2, K_rgb,
                                     tuple(map(int, kp2[m.trainIdx].pt)),
                                     depth_scale)
            if p_xyz is None or q_xyz is None:
                continue
            obj_pts.append(p_xyz)
            img_pts.append(kp2[m.trainIdx].pt)
            p2_pts.append(q_xyz)

        obj_pts = np.asarray(obj_pts, np.float32)
        p2_pts  = np.asarray(p2_pts,  np.float32)
        img_pts = np.asarray(img_pts, np.float32)
        if len(obj_pts) < 6:
            return obj_pts, p2_pts, None

    # PnP RANSAC pose estimation
    with profiler.timer("pnp_ransac"):
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj_pts, img_pts, K_rgb, None,
            iterationsCount=300, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_AP3P)
        if not ok:
            return obj_pts, p2_pts, None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3], T[:3, 3] = R, tvec.ravel()
    
    return obj_pts[inl.flatten()], p2_pts[inl.flatten()], T 