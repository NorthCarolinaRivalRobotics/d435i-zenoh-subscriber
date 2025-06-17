#!/usr/bin/env python3
"""
Camera calibration and configuration for RealSense D435i.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CameraCalibration:
    """Camera calibration data for RealSense D435i."""
    
    # Depth camera intrinsics
    K_depth: np.ndarray
    
    # RGB camera intrinsics  
    K_rgb: np.ndarray
    
    # Transform from depth to RGB camera
    T_depth_to_rgb: np.ndarray
    
    # Transform from RGB to depth camera (inverse)
    T_rgb_to_depth: np.ndarray
    
    @classmethod
    def create_default_d435i(cls) -> 'CameraCalibration':
        """Create default calibration for RealSense D435i."""
        
        K_depth = np.array([[387.31454, 0, 322.1206],
                            [0, 387.31454, 236.50139],
                            [0, 0, 1]], dtype=np.float32)
        
        K_rgb = np.array([[607.2676, 0, 316.65408],
                          [0, 607.149, 244.13338],
                          [0, 0, 1]], dtype=np.float32)
        
        # Rotation from depth to RGB
        R_d2r = np.array([[0.9999627,  -0.008320532,  0.0023323754],
                          [0.008310333,  0.999956,     0.0043491516],
                          [-0.00236846, -0.0043296064, 0.99998784]])
        
        # Translation from depth to RGB
        t_d2r = np.array([0.014476319, 0.0001452052, 0.00031550066])
        
        # Build 4x4 transform matrix
        T_d2r = np.eye(4)
        T_d2r[:3, :3], T_d2r[:3, 3] = R_d2r, t_d2r
        
        # Inverse transform (RGB to depth)
        T_r2d = np.linalg.inv(T_d2r)
        
        return cls(
            K_depth=K_depth,
            K_rgb=K_rgb,
            T_depth_to_rgb=T_d2r,
            T_rgb_to_depth=T_r2d
        )
    
    @classmethod
    def from_calibration_file(cls, filepath: str) -> 'CameraCalibration':
        """Load calibration from a file (future implementation)."""
        raise NotImplementedError("Loading from file not yet implemented")
    
    def save_to_file(self, filepath: str) -> None:
        """Save calibration to a file (future implementation)."""
        raise NotImplementedError("Saving to file not yet implemented") 