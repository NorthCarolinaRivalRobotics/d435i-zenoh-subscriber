#!/usr/bin/env python3
"""
Camera Data Manager - Easy integration of recording/playback functionality.

This module provides utilities to easily add recording and playback capabilities
to any script that uses the Zenoh D435i subscriber with minimal code changes.
"""
from __future__ import annotations

import argparse
import sys
import os
from typing import Optional, Union
import zenoh_d435i_subscriber as zd435i

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_source import create_data_source, CameraDataSource
from frame_reconstruction import PlaybackFrameData


class CameraDataManager:
    """
    High-level manager that provides easy integration of recording/playback.
    
    This class wraps the data source abstraction and provides a simple interface
    that can be used as a drop-in replacement for ZenohD435iSubscriber.
    """
    
    def __init__(self, record: bool = False, playback_file: Optional[str] = None,
                 recording_file: Optional[str] = None, loop: bool = True, realtime: bool = True):
        """
        Initialize the camera data manager.
        
        Args:
            record: Enable recording mode
            playback_file: Path to playback file (enables playback mode)
            recording_file: Path for recording file (optional, auto-generated if not provided)
            loop: Whether to loop playback (default: True)
            realtime: Whether to play back in real-time (default: True)
        """
        self.record = record
        self.playback_file = playback_file
        self.recording_file = recording_file
        self.loop = loop
        self.realtime = realtime
        
        # Create the appropriate data source
        self.data_source = create_data_source(
            record=record,
            playback_file=playback_file,
            recording_file=recording_file
        )
        
        # Determine mode for status reporting
        if playback_file:
            self.mode = "playback"
        elif record:
            self.mode = "recording"
        else:
            self.mode = "live"
    
    def connect(self) -> None:
        """Connect to the data source."""
        self.data_source.connect()
    
    def start_subscribing(self) -> None:
        """Start receiving/reading data."""
        self.data_source.start_subscribing()
    
    def get_latest_frames(self) -> Union[zd435i.PyFrameData, PlaybackFrameData]:
        """Get the latest frame data."""
        return self.data_source.get_latest_frames()
    
    def is_running(self) -> bool:
        """Check if the data source is active."""
        return self.data_source.is_running()
    
    def stop(self) -> None:
        """Stop the data source."""
        self.data_source.stop()
    
    def get_mode(self) -> str:
        """Get the current mode (live, recording, or playback)."""
        return self.mode
    
    def get_status_string(self) -> str:
        """Get a status string for display."""
        if self.mode == "playback":
            return f"[PLAYBACK] {os.path.basename(self.playback_file)}"
        elif self.mode == "recording":
            filename = os.path.basename(self.recording_file) if self.recording_file else "auto-generated"
            return f"[RECORDING] {filename}"
        else:
            return "[LIVE]"


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    """
    Add camera recording/playback arguments to an argument parser.
    
    Args:
        parser: The ArgumentParser to add arguments to
    """
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--record', action='store_true', 
                           help='Record camera data while processing')
    mode_group.add_argument('--playback', type=str, metavar='FILE',
                           help='Play back recorded data from file')
    
    # Recording options
    parser.add_argument('--recording-file', type=str, 
                       help='Path for recording file (default: auto-generated)')
    parser.add_argument('--loop', action='store_true', default=True,
                       help='Loop playback (default: True)')
    parser.add_argument('--no-loop', dest='loop', action='store_false',
                       help='Don\'t loop playback')
    parser.add_argument('--no-realtime', action='store_true',
                       help='Play back as fast as possible (not real-time)')


def create_camera_manager_from_args(args: argparse.Namespace) -> CameraDataManager:
    """
    Create a CameraDataManager from parsed command line arguments.
    
    Args:
        args: Parsed arguments from argparse containing camera options
        
    Returns:
        CameraDataManager configured according to the arguments
    """
    return CameraDataManager(
        record=args.record,
        playback_file=args.playback,
        recording_file=getattr(args, 'recording_file', None),
        loop=getattr(args, 'loop', True),
        realtime=not getattr(args, 'no_realtime', False)
    )


def setup_camera_manager(description: str = "Camera application") -> tuple[CameraDataManager, argparse.Namespace]:
    """
    Convenience function to set up argument parsing and create a camera manager.
    
    This is the easiest way to add recording/playback to an existing script.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        Tuple of (camera_manager, parsed_args)
        
    Example:
        camera_manager, args = setup_camera_manager("My camera app")
        camera_manager.connect()
        camera_manager.start_subscribing()
        
        while camera_manager.is_running():
            frame_data = camera_manager.get_latest_frames()
            # ... process frame_data as normal ...
    """
    parser = argparse.ArgumentParser(description=description)
    add_camera_args(parser)
    args = parser.parse_args()
    
    camera_manager = create_camera_manager_from_args(args)
    
    # Print mode information
    print(f"Camera mode: {camera_manager.get_status_string()}")
    
    return camera_manager, args


# Compatibility function for legacy code
def create_subscriber_with_args(args: argparse.Namespace) -> CameraDataManager:
    """
    Legacy compatibility function.
    
    Args:
        args: Parsed arguments
        
    Returns:
        CameraDataManager that can be used as a drop-in replacement for ZenohD435iSubscriber
    """
    return create_camera_manager_from_args(args) 