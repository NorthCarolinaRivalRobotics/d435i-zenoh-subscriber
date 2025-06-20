#!/usr/bin/env python3
"""
Computer vision and geometry utilities for RGB-D processing.
"""
from __future__ import annotations

import cv2
import numpy as np
from profiling import profiler
from collections import defaultdict

MINIMUM_VALID_DEPTH = 0.55
MAXIMUM_VALID_DEPTH = 7.0 

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


def estimate_multiframe_transform(frames_data: list,
                                K_rgb: np.ndarray, K_depth: np.ndarray,
                                T_rgb_depth: np.ndarray,
                                depth_scale: float = 1.0,
                                n_keypoints: int = 1000,
                                max_descriptor_distance: float = 30.0,
                                min_track_length: int = None):
    """
    Track features across N frames and estimate transform from frame 0 to frame N-1.
    Only uses features that are successfully tracked through ALL frames.
    
    Args:
        frames_data: List of (rgb, depth) tuples for N frames
        K_rgb: RGB camera intrinsic matrix
        K_depth: Depth camera intrinsic matrix  
        T_rgb_depth: Transform from depth to RGB camera
        depth_scale: Scale factor for depth values
        n_keypoints: Number of ORB keypoints to detect per frame
        max_descriptor_distance: Maximum distance for descriptor matching
        min_track_length: Minimum number of frames a feature must be tracked (default: all frames)
        
    Returns:
        tuple: (points_3d_frame0, points_3d_frameN, transform_matrix, num_tracked_features)
    """
    if len(frames_data) < 2:
        return np.array([]), np.array([]), None, 0
        
    N = len(frames_data)
    if min_track_length is None:
        min_track_length = N
        
    H, W = frames_data[0][0].shape[:2]
    
    # Register depth to RGB for all frames
    registered_depths = []
    for rgb, depth in frames_data:
        reg_depth = cv2.rgbd.registerDepth(
            K_depth.astype(np.float32),
            K_rgb.astype(np.float32),
            None,
            T_rgb_depth.astype(np.float32),
            depth.astype(np.float32),
            (int(W), int(H)),
            depthDilation=True,
        )
        registered_depths.append(reg_depth)
    
    # Detect ORB features in all frames
    orb = cv2.ORB_create(n_keypoints)
    keypoints_list = []
    descriptors_list = []
    
    for rgb, _ in frames_data:
        kp, des = orb.detectAndCompute(rgb, None)
        if des is None or len(des) == 0:
            return np.array([]), np.array([]), None, 0
        keypoints_list.append(kp)
        descriptors_list.append(des)
    
    # Match descriptors across consecutive frames to build tracks
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Build feature tracks: track_id -> [(frame_idx, keypoint_idx), ...]
    tracks = defaultdict(list)
    next_track_id = 0
    
    # Initialize tracks with frame 0
    for kp_idx in range(len(keypoints_list[0])):
        tracks[next_track_id].append((0, kp_idx))
        next_track_id += 1
    
    # For each subsequent frame, match with previous frame and extend tracks
    for frame_idx in range(1, N):
        try:
            matches = bf.match(descriptors_list[frame_idx-1], descriptors_list[frame_idx])
            matches = [m for m in matches if m.distance <= max_descriptor_distance]
            matches.sort(key=lambda m: m.distance)
            
            # Create mapping from previous frame keypoint to current frame keypoint
            prev_to_curr = {}
            for match in matches:
                prev_to_curr[match.queryIdx] = match.trainIdx
            
            # Extend existing tracks or create new ones
            tracks_to_extend = []
            for track_id, track_points in tracks.items():
                # Get the most recent point in this track
                if track_points and track_points[-1][0] == frame_idx - 1:
                    prev_kp_idx = track_points[-1][1]
                    if prev_kp_idx in prev_to_curr:
                        tracks_to_extend.append((track_id, prev_to_curr[prev_kp_idx]))
            
            # Extend tracks
            for track_id, curr_kp_idx in tracks_to_extend:
                tracks[track_id].append((frame_idx, curr_kp_idx))
                
        except Exception as e:
            print(f"Error matching frame {frame_idx-1} to {frame_idx}: {e}")
            continue
    
    # Filter tracks that span the required minimum length
    valid_tracks = {}
    for track_id, track_points in tracks.items():
        if len(track_points) >= min_track_length:
            # Ensure track starts at frame 0 and spans to the last frame
            frame_indices = [pt[0] for pt in track_points]
            if 0 in frame_indices and (N-1) in frame_indices:
                valid_tracks[track_id] = track_points
    
    print(f"Found {len(valid_tracks)} features tracked across {min_track_length}+ frames")
    
    if len(valid_tracks) < 20:
        return np.array([]), np.array([]), None, len(valid_tracks)
    
    # Extract 3D points for frame 0 and frame N-1 from valid tracks
    obj_pts, img_pts = [], []
    
    for track_id, track_points in valid_tracks.items():
        # Find points in frame 0 and frame N-1
        frame0_point = None
        frameN_point = None
        
        for frame_idx, kp_idx in track_points:
            if frame_idx == 0:
                frame0_point = (frame_idx, kp_idx)
            elif frame_idx == N-1:
                frameN_point = (frame_idx, kp_idx)
        
        if frame0_point is None or frameN_point is None:
            continue
            
        # Get 3D point from frame 0
        kp0 = keypoints_list[frame0_point[0]][frame0_point[1]]
        xyz0 = rgb_pixel_to_xyz(registered_depths[0], K_rgb,
                               tuple(map(int, kp0.pt)), depth_scale)
        
        # Get 2D point from frame N-1 for PnP
        kpN = keypoints_list[frameN_point[0]][frameN_point[1]]
        
        if xyz0 is None:
            continue
        if xyz0[2] < MINIMUM_VALID_DEPTH or xyz0[2] > MAXIMUM_VALID_DEPTH:
            continue
            
        obj_pts.append(xyz0)
        img_pts.append(kpN.pt)
    
    obj_pts = np.asarray(obj_pts, np.float32)
    img_pts = np.asarray(img_pts, np.float32)
    
    print(f"Using {len(obj_pts)} valid 3D-2D correspondences for PnP")
    
    if len(obj_pts) < 20:
        return obj_pts, np.array([]), None, len(valid_tracks)
    
    # Solve PnP RANSAC for transform from frame 0 to frame N-1
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K_rgb, None,
            iterationsCount=500, reprojectionError=2.0,
            confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
        
        if not ok or inliers is None:
            return obj_pts, np.array([]), None, len(valid_tracks)
        
        # Convert to 4x4 transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        
        # Extract inlier points for visualization
        inlier_indices = inliers.flatten()
        obj_pts_inliers = obj_pts[inlier_indices]
        
        # Get corresponding 3D points from frame N-1 for visualization
        frameN_3d_points = []
        inlier_count = 0
        for track_id, track_points in valid_tracks.items():
            if inlier_count >= len(inlier_indices):
                break
            if inlier_count in inlier_indices:
                # Find frame N-1 point
                for frame_idx, kp_idx in track_points:
                    if frame_idx == N-1:
                        kpN = keypoints_list[frame_idx][kp_idx]
                        xyzN = rgb_pixel_to_xyz(registered_depths[N-1], K_rgb,
                                              tuple(map(int, kpN.pt)), depth_scale)
                        if xyzN is not None:
                            frameN_3d_points.append(xyzN)
                        break
            inlier_count += 1
            
        frameN_3d_points = np.asarray(frameN_3d_points, np.float32)
        
        print(f"PnP RANSAC successful: {len(inlier_indices)} inliers out of {len(obj_pts)} correspondences")
        
        return obj_pts_inliers, frameN_3d_points, T, len(valid_tracks)
        
    except Exception as e:
        print(f"Error in PnP solving: {e}")
        return obj_pts, np.array([]), None, len(valid_tracks)


def estimate_frame_transform(rgb1: np.ndarray, depth1: np.ndarray,
                             rgb2: np.ndarray, depth2: np.ndarray,
                             K_rgb: np.ndarray, K_depth: np.ndarray,
                             T_rgb_depth: np.ndarray,
                             depth_scale: float = 1.0,
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
    orb = cv2.ORB_create(n_keypoints)
    kp1, des1 = orb.detectAndCompute(rgb1, None)
    kp2, des2 = orb.detectAndCompute(rgb2, None)
    
    # Check if descriptors were found
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return np.array([]), np.array([]), None
    
    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
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
        if p_xyz[2] < MINIMUM_VALID_DEPTH or p_xyz[2] > MAXIMUM_VALID_DEPTH:
            continue
        if q_xyz[2] < MINIMUM_VALID_DEPTH or q_xyz[2] > MAXIMUM_VALID_DEPTH:
            continue
        obj_pts.append(p_xyz)
        img_pts.append(kp2[m.trainIdx].pt)
        p2_pts.append(q_xyz)

    obj_pts = np.asarray(obj_pts, np.float32)
    p2_pts  = np.asarray(p2_pts,  np.float32)
    img_pts = np.asarray(img_pts, np.float32)
    if len(obj_pts) < 20:
        return obj_pts, p2_pts, None

    # PnP RANSAC pose estimation
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


# CAMERA_TO_ROBOT_FRAME = np.array([[ 0.0,  0.0,  1.0,  0.0],  # Robot X = Camera Z (forward)
#                                   [-1.0,  0.0,  0.0,  0.0],  # Robot Y = -Camera X (right→left)
#                                   [ 0.0, -1.0,  0.0,  0.0],  # Robot Z = -Camera Y (down→up)
#                                   [ 0.0,  0.0,  0.0,  1.0]])

CAMERA_TO_ROBOT_FRAME = np.array([[ 0.0,  0.0,  1.0,  0.0],
                                  [0.0,  1.0,  0.0,  0.0], 
                                  [ 1.0, 0.0,  0.0,  0.0],  
                                  [ 0.0,  0.0,  0.0,  1.0]])