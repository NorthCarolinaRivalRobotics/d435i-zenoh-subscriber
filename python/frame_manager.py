#!/usr/bin/env python3
"""
Optimized frame management for efficient Zenoh D435i frame acquisition.
"""
from __future__ import annotations

import time
import threading
import queue
from typing import Optional, Tuple
import zenoh_d435i_subscriber as zd435i
from profiling import profiler


class OptimizedFrameManager:
    """
    Optimized frame manager with background frame acquisition and buffering.
    """
    
    def __init__(self, buffer_size: int = 3, acquisition_timeout: float = 0.1):
        """
        Initialize the frame manager.
        
        Args:
            buffer_size: Number of frames to buffer (keep small to reduce latency)
            acquisition_timeout: Timeout for frame acquisition attempts
        """
        self.buffer_size = buffer_size
        self.acquisition_timeout = acquisition_timeout
        
        # Zenoh subscriber
        self.subscriber = zd435i.ZenohD435iSubscriber()
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.acquisition_thread = None
        self.running = False
        
        # Frame tracking
        self.last_frame_id = -1
        self.frames_received = 0
        self.frames_dropped = 0
        
        # Statistics
        self.acquisition_times = []
        
    def connect_and_start(self) -> None:
        """Connect to Zenoh and start background frame acquisition."""
        with profiler.timer("zenoh_connection"):
            self.subscriber.connect()
            self.subscriber.start_subscribing()
        
        # Start background acquisition thread
        self.running = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()
        
        print(f"Frame manager started with buffer size {self.buffer_size}")
    
    def _acquisition_loop(self) -> None:
        """Background thread for continuous frame acquisition."""
        consecutive_empty = 0
        
        while self.running:
            try:
                start_time = time.perf_counter()
                
                # Get latest frames from Zenoh
                fd = self.subscriber.get_latest_frames()
                
                acquisition_time = time.perf_counter() - start_time
                self.acquisition_times.append(acquisition_time)
                
                # Keep only recent timing data
                if len(self.acquisition_times) > 100:
                    self.acquisition_times = self.acquisition_times[-50:]
                
                if fd.frame_count == 0:
                    consecutive_empty += 1
                    # Adaptive sleep based on how long we've been waiting
                    sleep_time = min(0.001 * consecutive_empty, 0.01)  # 1ms to 10ms
                    time.sleep(sleep_time)
                    continue
                
                consecutive_empty = 0
                
                # Skip if it's the same frame we already processed
                if fd.frame_count == self.last_frame_id:
                    time.sleep(0.001)  # Short sleep to avoid busy waiting
                    continue
                
                self.last_frame_id = fd.frame_count
                self.frames_received += 1
                
                # Try to add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((fd, time.perf_counter()))
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass
                    
                    try:
                        self.frame_queue.put_nowait((fd, time.perf_counter()))
                    except queue.Full:
                        self.frames_dropped += 1
                        
            except Exception as e:
                print(f"Error in acquisition loop: {e}")
                time.sleep(0.01)
    
    def get_latest_frame(self) -> Optional[Tuple[any, float]]:
        """
        Get the latest available frame (non-blocking).
        
        Returns:
            Tuple of (frame_data, timestamp) or None if no frame available
        """
        try:
            # Get the most recent frame, discarding older ones
            latest_frame = None
            while True:
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            return latest_frame
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def wait_for_new_frame(self, timeout: float = 0.1) -> Optional[Tuple[any, float]]:
        """
        Wait for a new frame with timeout.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Tuple of (frame_data, timestamp) or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            print(f"Error waiting for frame: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get acquisition statistics."""
        if not self.acquisition_times:
            return {}
        
        return {
            'frames_received': self.frames_received,
            'frames_dropped': self.frames_dropped,
            'queue_size': self.frame_queue.qsize(),
            'avg_acquisition_ms': sum(self.acquisition_times) / len(self.acquisition_times) * 1000,
            'max_acquisition_ms': max(self.acquisition_times) * 1000,
            'drop_rate': self.frames_dropped / max(1, self.frames_received + self.frames_dropped)
        }
    
    def stop(self) -> None:
        """Stop frame acquisition and cleanup."""
        self.running = False
        
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        
        self.subscriber.stop()
        
        # Print final stats
        stats = self.get_stats()
        if stats:
            print(f"\nFrame Manager Stats:")
            print(f"  Frames received: {stats['frames_received']}")
            print(f"  Frames dropped: {stats['frames_dropped']}")
            print(f"  Drop rate: {stats['drop_rate']:.1%}")
            print(f"  Avg acquisition: {stats['avg_acquisition_ms']:.1f}ms")
            print(f"  Max acquisition: {stats['max_acquisition_ms']:.1f}ms")
    
    def is_running(self) -> bool:
        """Check if the frame manager is running."""
        return self.running and self.subscriber.is_running() 