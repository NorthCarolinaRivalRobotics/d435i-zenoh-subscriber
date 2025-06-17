#!/usr/bin/env python3
"""
Frame reconstruction utilities for playback.
Works around limitations of creating Rust frame objects from Python.
"""
from __future__ import annotations

import numpy as np
import zenoh_d435i_subscriber as zd435i
from typing import Optional, Dict, Any


class PlaybackFrameData:
    """Frame data container for playback that mimics PyFrameData interface."""
    
    def __init__(self, frame_record: Dict[str, Any]):
        self.frame_count = frame_record['frame_count']
        self.timestamp = frame_record['timestamp']
        
        # Store raw data for reconstruction
        self._rgb_data = frame_record['rgb']
        self._depth_data = frame_record['depth'] 
        self._motion_data = frame_record['motion']
        
        # Lazy-loaded properties
        self._rgb_frame = None
        self._depth_frame = None
        self._motion_frame = None
    
    @property
    def rgb(self) -> Optional['PlaybackRgbFrame']:
        """Get RGB frame data."""
        if self._rgb_data is None:
            return None
        if self._rgb_frame is None:
            self._rgb_frame = PlaybackRgbFrame(self._rgb_data)
        return self._rgb_frame
    
    @property
    def depth(self) -> Optional['PlaybackDepthFrame']:
        """Get depth frame data."""
        if self._depth_data is None:
            return None
        if self._depth_frame is None:
            self._depth_frame = PlaybackDepthFrame(self._depth_data)
        return self._depth_frame
    
    @property
    def motion(self) -> Optional['PlaybackMotionFrame']:
        """Get motion frame data."""
        if self._motion_data is None:
            return None
        if self._motion_frame is None:
            self._motion_frame = PlaybackMotionFrame(self._motion_data)
        return self._motion_frame


class PlaybackRgbFrame:
    """RGB frame data for playback that mimics PyRgbFrame interface."""
    
    def __init__(self, rgb_data: Dict[str, Any]):
        self.timestamp = rgb_data['timestamp']
        self.width = rgb_data['width']
        self.height = rgb_data['height']
        
        # Store the raw RGB data
        self._raw_data = rgb_data['data']
        
        # Convert raw RGB data to JPEG format for compatibility with cv2.imdecode()
        self._jpeg_data = None
    
    def get_data(self) -> bytes:
        """Get RGB data as JPEG bytes for compatibility with cv2.imdecode()."""
        if self._jpeg_data is None:
            import cv2
            import numpy as np
            
            # Convert raw RGB bytes to numpy array
            # Raw data is RGB format (width * height * 3 bytes)
            raw_array = np.frombuffer(self._raw_data, dtype=np.uint8)
            
            # Reshape to (height, width, 3) - RGB format
            rgb_image = raw_array.reshape((self.height, self.width, 3))
            
            # Convert RGB to BGR for OpenCV (since cv2.imdecode expects BGR)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            success, jpeg_data = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                self._jpeg_data = jpeg_data.tobytes()
            else:
                raise RuntimeError("Failed to encode RGB data as JPEG")
        
        return self._jpeg_data


class PlaybackDepthFrame:
    """Depth frame data for playback that mimics PyDepthFrame interface."""
    
    # Constants from Rust types.rs
    DEPTH_SCALE_FACTOR = 8738
    MINIMUM_DISTANCE_METERS = 0.5
    
    def __init__(self, depth_data: Dict[str, Any]):
        self.raw_data = depth_data['raw_data']
        self.timestamp = depth_data['timestamp']
        self.width = depth_data['width']
        self.height = depth_data['height']
        self._converted_data = None
    
    def _decode_u16_to_meters(self, code: int) -> float:
        """Convert u16 depth code to meters using the same formula as the Rust code."""
        return (code / self.DEPTH_SCALE_FACTOR) + self.MINIMUM_DISTANCE_METERS
    
    def get_data_2d(self) -> np.ndarray:
        """Get depth data as 2D numpy array in meters."""
        if self._converted_data is None:
            # Convert from raw u16 data to meters (float32)
            raw_array = np.array(self.raw_data, dtype=np.uint16)
            # Apply the depth conversion formula
            meters_array = (raw_array.astype(np.float32) / self.DEPTH_SCALE_FACTOR) + self.MINIMUM_DISTANCE_METERS
            self._converted_data = meters_array.reshape((self.height, self.width))
        return self._converted_data
    
    def get_data(self) -> np.ndarray:
        """Get depth data as 1D numpy array in meters."""
        if self._converted_data is None:
            # Convert from raw u16 data to meters (float32)
            raw_array = np.array(self.raw_data, dtype=np.uint16)
            # Apply the depth conversion formula
            self._converted_data = (raw_array.astype(np.float32) / self.DEPTH_SCALE_FACTOR) + self.MINIMUM_DISTANCE_METERS
        else:
            self._converted_data = self._converted_data.flatten()
        return self._converted_data
    
    def get_raw_data(self) -> np.ndarray:
        """Get raw u16 depth data (for compatibility with recording system)."""
        return np.array(self.raw_data, dtype=np.uint16)


class PlaybackMotionFrame:
    """Motion frame data for playback that mimics PyMotionFrame interface."""
    
    def __init__(self, motion_data: Dict[str, Any]):
        self.gyro = motion_data['gyro']
        self.accel = motion_data['accel']
        self.timestamp = motion_data['timestamp'] 