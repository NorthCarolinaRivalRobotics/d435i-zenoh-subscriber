#!/usr/bin/env python3
"""
Abstract data source interface for camera data with recording/playback capabilities.
"""
from __future__ import annotations

import os
import time
import pickle
import gzip
from abc import ABC, abstractmethod
from typing import Optional, Union
import zenoh_d435i_subscriber as zd435i
from frame_reconstruction import PlaybackFrameData


class CameraDataSource(ABC):
    """Abstract base class for camera data sources."""
    
    @abstractmethod
    def connect(self) -> None:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    def start_subscribing(self) -> None:
        """Start receiving/reading data."""
        pass
    
    @abstractmethod
    def get_latest_frames(self) -> Union[zd435i.PyFrameData, PlaybackFrameData]:
        """Get the latest frame data."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the data source is active."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the data source."""
        pass


class LiveCameraDataSource(CameraDataSource):
    """Live camera data source using Zenoh subscriber."""
    
    def __init__(self):
        self.subscriber = zd435i.ZenohD435iSubscriber()
    
    def connect(self) -> None:
        """Connect to Zenoh."""
        self.subscriber.connect()
    
    def start_subscribing(self) -> None:
        """Start subscribing to camera data."""
        self.subscriber.start_subscribing()
    
    def get_latest_frames(self) -> zd435i.PyFrameData:
        """Get latest frames from Zenoh."""
        return self.subscriber.get_latest_frames()
    
    def is_running(self) -> bool:
        """Check if subscriber is running."""
        return self.subscriber.is_running()
    
    def stop(self) -> None:
        """Stop the subscriber."""
        self.subscriber.stop()


class RecordingCameraDataSource(CameraDataSource):
    """Recording camera data source that saves frames to disk while streaming live data."""
    
    def __init__(self, recording_path: str):
        self.subscriber = zd435i.ZenohD435iSubscriber()
        self.recording_path = recording_path
        self.recording_file = None
        self.frame_count = 0
        self.start_time = None
        
        # Create recording directory if it doesn't exist
        recording_dir = os.path.dirname(recording_path)
        if recording_dir:  # Only create directory if there's actually a directory path
            os.makedirs(recording_dir, exist_ok=True)
        
        print(f"Recording will be saved to: {recording_path}")
    
    def connect(self) -> None:
        """Connect to Zenoh and open recording file."""
        self.subscriber.connect()
        self.recording_file = gzip.open(self.recording_path, 'wb')
        self.start_time = time.time()
        print("Recording started")
    
    def start_subscribing(self) -> None:
        """Start subscribing to camera data."""
        self.subscriber.start_subscribing()
    
    def get_latest_frames(self) -> zd435i.PyFrameData:
        """Get latest frames and record them."""
        frame_data = self.subscriber.get_latest_frames()
        
        # Record frame if it has data
        if frame_data.frame_count > 0 and self.recording_file is not None:
            self._record_frame(frame_data)
        
        return frame_data
    
    def _record_frame(self, frame_data: zd435i.PyFrameData) -> None:
        """Record a frame to disk."""
        try:
            # Create a serializable version of the frame
            frame_record = {
                'timestamp': time.time() - self.start_time,  # Relative timestamp
                'frame_count': frame_data.frame_count,
                'rgb': None,
                'depth': None,
                'motion': None
            }
            
            # Extract RGB data if available
            if frame_data.rgb is not None:
                frame_record['rgb'] = {
                    'data': bytes(frame_data.rgb.get_data()),
                    'timestamp': frame_data.rgb.timestamp,
                    'width': frame_data.rgb.width,
                    'height': frame_data.rgb.height
                }
            
            # Extract depth data if available
            if frame_data.depth is not None:
                # Use get_raw_data() method to get the raw u16 data
                raw_depth_array = frame_data.depth.get_raw_data()
                frame_record['depth'] = {
                    'raw_data': raw_depth_array.tolist(),  # Convert numpy array to list for serialization
                    'timestamp': frame_data.depth.timestamp,
                    'width': frame_data.depth.width,
                    'height': frame_data.depth.height
                }
            
            # Extract motion data if available
            if frame_data.motion is not None:
                frame_record['motion'] = {
                    'gyro': frame_data.motion.gyro,
                    'accel': frame_data.motion.accel,
                    'timestamp': frame_data.motion.timestamp
                }
            
            # Write to compressed file
            pickle.dump(frame_record, self.recording_file)
            self.frame_count += 1
            
            # Print recording status every 100 frames
            if self.frame_count % 100 == 0:
                duration = time.time() - self.start_time
                print(f"Recorded {self.frame_count} frames in {duration:.1f}s "
                      f"({self.frame_count/duration:.1f} fps)")
                
        except Exception as e:
            print(f"Error recording frame: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
    
    def is_running(self) -> bool:
        """Check if subscriber is running."""
        return self.subscriber.is_running()
    
    def stop(self) -> None:
        """Stop recording and close file."""
        self.subscriber.stop()
        if self.recording_file is not None:
            self.recording_file.close()
            self.recording_file = None
            duration = time.time() - self.start_time if self.start_time else 0
            print(f"Recording stopped. Saved {self.frame_count} frames in {duration:.1f}s to {self.recording_path}")


class PlaybackCameraDataSource(CameraDataSource):
    """Playback camera data source that reads recorded frames from disk."""
    
    def __init__(self, recording_path: str, loop: bool = True, realtime: bool = True):
        self.recording_path = recording_path
        self.loop = loop
        self.realtime = realtime
        self.recording_file = None
        self.frames = []
        self.current_frame_index = 0
        self.start_time = None
        self.playback_start_time = None
        self.running = False
        self.last_frame_id = -1
        
        if not os.path.exists(recording_path):
            raise FileNotFoundError(f"Recording file not found: {recording_path}")
        
        print(f"Playback will read from: {recording_path}")
    
    def connect(self) -> None:
        """Load recording file."""
        print("Loading recorded frames...")
        start_load = time.time()
        
        with gzip.open(self.recording_path, 'rb') as f:
            try:
                while True:
                    frame = pickle.load(f)
                    self.frames.append(frame)
            except EOFError:
                pass  # End of file reached
        
        load_time = time.time() - start_load
        print(f"Loaded {len(self.frames)} frames in {load_time:.1f}s")
        
        if len(self.frames) == 0:
            raise ValueError("No frames found in recording file")
    
    def start_subscribing(self) -> None:
        """Start playback."""
        self.running = True
        self.start_time = time.time()
        self.playback_start_time = time.time()
        self.current_frame_index = 0
        self.last_frame_id = -1
        print("Playback started")
    
    def get_latest_frames(self) -> PlaybackFrameData:
        """Get frame based on playback timing."""
        if not self.running or len(self.frames) == 0:
            # Return empty frame data with frame_count=0 to indicate no data
            empty_record = {
                'timestamp': 0,
                'frame_count': 0,
                'rgb': None,
                'depth': None,
                'motion': None
            }
            return PlaybackFrameData(empty_record)
        
        # Handle real-time playback
        if self.realtime:
            current_time = time.time() - self.playback_start_time
            
            # Find the appropriate frame based on timestamp
            target_index = self.current_frame_index
            for i in range(self.current_frame_index, len(self.frames)):
                if self.frames[i]['timestamp'] <= current_time:
                    target_index = i
                else:
                    break
            
            self.current_frame_index = target_index
        
        # Get current frame
        if self.current_frame_index >= len(self.frames):
            if self.loop:
                self.current_frame_index = 0
                self.playback_start_time = time.time()  # Reset timing for loop
                self.last_frame_id = -1
                print("Playback looped")
            else:
                self.running = False
                print("Playback finished")
                empty_record = {
                    'timestamp': 0,
                    'frame_count': 0,
                    'rgb': None,
                    'depth': None,
                    'motion': None
                }
                return PlaybackFrameData(empty_record)
        
        frame_record = self.frames[self.current_frame_index]
        
        # Check if this is a new frame (to simulate the live API behavior)
        if frame_record['frame_count'] == self.last_frame_id:
            # Return the same frame data but don't advance
            return PlaybackFrameData(frame_record)
        
        self.last_frame_id = frame_record['frame_count']
        
        # Advance frame index for next call (only in non-realtime mode)
        if not self.realtime:
            self.current_frame_index += 1
        
        return PlaybackFrameData(frame_record)
    
    def is_running(self) -> bool:
        """Check if playback is active."""
        return self.running
    
    def stop(self) -> None:
        """Stop playback."""
        self.running = False
        print("Playback stopped")


def create_data_source(record: bool = False, playback_file: Optional[str] = None, 
                      recording_file: Optional[str] = None) -> CameraDataSource:
    """Factory function to create the appropriate data source."""
    
    if playback_file is not None:
        return PlaybackCameraDataSource(playback_file)
    elif record:
        if recording_file is None:
            # Generate default recording filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            recording_file = f"recordings/camera_data_{timestamp}.pkl.gz"
        return RecordingCameraDataSource(recording_file)
    else:
        return LiveCameraDataSource() 