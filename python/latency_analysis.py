#!/usr/bin/env python3
"""
Latency analysis tool for the Zenoh D435i subscriber.

This script helps identify where the 0.5s latency is coming from by measuring
each component of the pipeline separately.
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import zenoh_d435i_subscriber as zd435i
from collections import deque
import statistics

from camera_config import CameraCalibration
from vision_utils import estimate_frame_transform
from visualization import RerunVisualizer


class LatencyAnalyzer:
    """Analyzes latency in different parts of the pipeline."""
    
    def __init__(self):
        self.zenoh_times = deque(maxlen=100)
        self.decode_times = deque(maxlen=100)
        self.rerun_times = deque(maxlen=100)
        self.feature_times = deque(maxlen=100)
        self.total_times = deque(maxlen=100)
        
        # Frame timing
        self.frame_receive_times = deque(maxlen=100)
        self.frame_ids = deque(maxlen=100)
        
    def analyze_pipeline(self, duration_seconds: int = 30):
        """Run comprehensive latency analysis."""
        print(f"Starting {duration_seconds}s latency analysis...")
        print("This will measure each component of the pipeline separately.")
        
        # Setup
        visualizer = RerunVisualizer("latency_analysis", spawn=True)
        camera_cal = CameraCalibration.create_default_d435i()
        sub = zd435i.ZenohD435iSubscriber()
        sub.connect()
        sub.start_subscribing()
        
        last_frame = None
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                pipeline_start = time.perf_counter()
                
                # 1. Measure Zenoh frame acquisition
                zenoh_start = time.perf_counter()
                fd = sub.get_latest_frames()
                zenoh_time = time.perf_counter() - zenoh_start
                self.zenoh_times.append(zenoh_time)
                
                if fd.frame_count == 0:
                    time.sleep(0.001)
                    continue
                    
                # Track frame receive timing
                current_time = time.time()
                self.frame_receive_times.append(current_time)
                self.frame_ids.append(fd.frame_count)
                
                # 2. Measure image decoding
                decode_start = time.perf_counter()
                rgb_buf = fd.rgb.get_data()
                w, h = fd.rgb.width, fd.rgb.height
                
                rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                if rgb_bgr is None:
                    rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))
                    
                rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                depth_img = fd.depth.get_data_2d().astype(np.float32)
                decode_time = time.perf_counter() - decode_start
                self.decode_times.append(decode_time)
                
                # 3. Measure Rerun visualization
                rerun_start = time.perf_counter()
                visualizer.log_rgb_image(rgb_img)
                visualizer.log_depth_image(depth_img, meter=1.0)
                rerun_time = time.perf_counter() - rerun_start
                self.rerun_times.append(rerun_time)
                
                # 4. Measure feature matching (if we have previous frame)
                feature_time = 0
                if last_frame is not None:
                    feature_start = time.perf_counter()
                    P1, P2, T = estimate_frame_transform(
                        last_frame["rgb"], last_frame["depth"],
                        rgb_img, depth_img,
                        camera_cal.K_rgb, camera_cal.K_depth,
                        camera_cal.T_rgb_to_depth,
                        depth_scale=1.0
                    )
                    
                    if len(P1) > 0:
                        visualizer.log_3d_matches(P1, P2)
                    if T is not None:
                        visualizer.log_camera_pose(T, frame_count)
                        
                    feature_time = time.perf_counter() - feature_start
                    self.feature_times.append(feature_time)
                
                # Total pipeline time
                total_time = time.perf_counter() - pipeline_start
                self.total_times.append(total_time)
                
                last_frame = {"rgb": rgb_img, "depth": depth_img}
                frame_count += 1
                
                # Print progress every 5 seconds
                elapsed = time.time() - start_time
                if frame_count % 50 == 0:
                    progress = (elapsed / duration_seconds) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count} frames)")
                    
        except KeyboardInterrupt:
            print("Analysis interrupted by user")
        finally:
            sub.stop()
            
        self.print_analysis()
        self.analyze_frame_timing()
        
    def print_analysis(self):
        """Print detailed latency analysis."""
        print("\n" + "="*60)
        print("LATENCY ANALYSIS RESULTS")
        print("="*60)
        
        components = [
            ("Zenoh Frame Acquisition", self.zenoh_times),
            ("Image Decoding", self.decode_times),
            ("Rerun Visualization", self.rerun_times),
            ("Feature Matching", self.feature_times),
            ("Total Pipeline", self.total_times)
        ]
        
        print(f"{'Component':<25} {'Count':<8} {'Mean(ms)':<10} {'Max(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10}")
        print("-" * 70)
        
        for name, times in components:
            if not times:
                continue
                
            times_ms = [t * 1000 for t in times]
            mean_ms = statistics.mean(times_ms)
            max_ms = max(times_ms)
            p95_ms = np.percentile(times_ms, 95) if times_ms else 0
            p99_ms = np.percentile(times_ms, 99) if times_ms else 0
            
            print(f"{name:<25} {len(times):<8} {mean_ms:<10.1f} {max_ms:<10.1f} {p95_ms:<10.1f} {p99_ms:<10.1f}")
        
        # Calculate percentages
        if self.total_times:
            total_mean = statistics.mean(self.total_times) * 1000
            print(f"\nComponent breakdown (% of total {total_mean:.1f}ms):")
            
            for name, times in components[:-1]:  # Exclude total pipeline
                if times:
                    mean_ms = statistics.mean(times) * 1000
                    percentage = (mean_ms / total_mean) * 100
                    print(f"  {name:<25} {percentage:>6.1f}%")
                    
    def analyze_frame_timing(self):
        """Analyze frame arrival timing and potential buffering issues."""
        print("\n" + "="*60)
        print("FRAME TIMING ANALYSIS")
        print("="*60)
        
        if len(self.frame_receive_times) < 2:
            print("Not enough frames to analyze timing")
            return
            
        # Calculate inter-frame intervals
        intervals = []
        for i in range(1, len(self.frame_receive_times)):
            interval = self.frame_receive_times[i] - self.frame_receive_times[i-1]
            intervals.append(interval * 1000)  # Convert to ms
            
        if intervals:
            mean_interval = statistics.mean(intervals)
            fps = 1000 / mean_interval if mean_interval > 0 else 0
            max_interval = max(intervals)
            min_interval = min(intervals)
            
            print(f"Frame arrival rate: {fps:.1f} FPS")
            print(f"Mean interval: {mean_interval:.1f}ms")
            print(f"Min interval: {min_interval:.1f}ms")
            print(f"Max interval: {max_interval:.1f}ms")
            
            # Check for frame ID gaps (indicating dropped frames)
            frame_gaps = []
            for i in range(1, len(self.frame_ids)):
                gap = self.frame_ids[i] - self.frame_ids[i-1] - 1
                if gap > 0:
                    frame_gaps.append(gap)
                    
            if frame_gaps:
                total_dropped = sum(frame_gaps)
                print(f"Dropped frames: {total_dropped} ({len(frame_gaps)} gaps)")
            else:
                print("No dropped frames detected")
                
            # Look for buffering issues (long intervals)
            long_intervals = [i for i in intervals if i > mean_interval * 2]
            if long_intervals:
                print(f"Potential buffering events: {len(long_intervals)} (intervals > {mean_interval*2:.1f}ms)")
                
    def diagnose_latency(self):
        """Provide specific recommendations based on analysis."""
        print("\n" + "="*60)
        print("LATENCY DIAGNOSIS & RECOMMENDATIONS")
        print("="*60)
        
        if not self.total_times:
            print("No data to analyze")
            return
            
        # Identify primary bottleneck
        mean_times = {}
        if self.zenoh_times:
            mean_times["zenoh"] = statistics.mean(self.zenoh_times) * 1000
        if self.decode_times:
            mean_times["decode"] = statistics.mean(self.decode_times) * 1000
        if self.rerun_times:
            mean_times["rerun"] = statistics.mean(self.rerun_times) * 1000
        if self.feature_times:
            mean_times["features"] = statistics.mean(self.feature_times) * 1000
            
        if mean_times:
            bottleneck = max(mean_times, key=mean_times.get)
            bottleneck_time = mean_times[bottleneck]
            total_mean = statistics.mean(self.total_times) * 1000
            
            print(f"Primary bottleneck: {bottleneck} ({bottleneck_time:.1f}ms, {(bottleneck_time/total_mean)*100:.1f}% of total)")
            
            # Provide specific recommendations
            if bottleneck == "features":
                print("\nRECOMMENDATIONS:")
                print("1. Move feature matching to background thread (see alignment_and_matching_threaded.py)")
                print("2. Reduce feature matching frequency (every 2-3 frames)")
                print("3. Reduce number of ORB keypoints")
                print("4. Consider faster feature detectors (FAST, ORB with fewer features)")
                
            elif bottleneck == "rerun":
                print("\nRECOMMENDATIONS:")
                print("1. Check Rerun viewer performance - try closing/reopening")
                print("2. Reduce image resolution for visualization")
                print("3. Log data less frequently")
                print("4. Use Rerun's batching APIs if available")
                
            elif bottleneck == "decode":
                print("\nRECOMMENDATIONS:")
                print("1. The Rust decode should be fast - check if JPEG quality is too high")
                print("2. Consider reducing image resolution at source")
                print("3. Profile the depth decompression (zstd)")
                
            elif bottleneck == "zenoh":
                print("\nRECOMMENDATIONS:")
                print("1. Check network latency to Zenoh publisher")
                print("2. Verify Zenoh configuration for low latency")
                print("3. Check if publisher is keeping up with target framerate")


def main():
    """Run latency analysis."""
    analyzer = LatencyAnalyzer()
    
    try:
        duration = 30  # Analyze for 30 seconds
        analyzer.analyze_pipeline(duration)
        analyzer.diagnose_latency()
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted")
    
    print("\nLatency analysis complete!")
    print("Try the optimized versions:")
    print("  - alignment_and_matching.py (reduced feature matching)")
    print("  - alignment_and_matching_threaded.py (background processing)")


if __name__ == "__main__":
    main() 