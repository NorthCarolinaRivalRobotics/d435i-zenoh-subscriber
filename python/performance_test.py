#!/usr/bin/env python3
"""
Performance comparison test for frame acquisition optimizations.
"""
from __future__ import annotations

import time
import statistics
import zenoh_d435i_subscriber as zd435i
from frame_manager import OptimizedFrameManager


def test_original_acquisition(duration_seconds: int = 30) -> dict:
    """Test the original frame acquisition method."""
    print("Testing original frame acquisition...")
    
    sub = zd435i.ZenohD435iSubscriber()
    sub.connect()
    sub.start_subscribing()
    
    acquisition_times = []
    frame_count = 0
    last_frame_id = -1
    
    start_time = time.perf_counter()
    
    try:
        while time.perf_counter() - start_time < duration_seconds:
            acq_start = time.perf_counter()
            
            fd = sub.get_latest_frames()
            
            acq_time = time.perf_counter() - acq_start
            acquisition_times.append(acq_time)
            
            if fd.frame_count == 0:
                time.sleep(0.05)
                continue
            if fd.frame_count == last_frame_id:
                time.sleep(0.02)
                continue
                
            last_frame_id = fd.frame_count
            frame_count += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        sub.stop()
    
    total_time = time.perf_counter() - start_time
    
    return {
        'method': 'original',
        'duration': total_time,
        'frames_processed': frame_count,
        'fps': frame_count / total_time,
        'avg_acquisition_ms': statistics.mean(acquisition_times) * 1000,
        'max_acquisition_ms': max(acquisition_times) * 1000,
        'median_acquisition_ms': statistics.median(acquisition_times) * 1000
    }


def test_optimized_acquisition(duration_seconds: int = 30) -> dict:
    """Test the optimized frame acquisition method."""
    print("Testing optimized frame acquisition...")
    
    frame_manager = OptimizedFrameManager(buffer_size=2)
    frame_manager.connect_and_start()
    
    acquisition_times = []
    frame_count = 0
    
    start_time = time.perf_counter()
    
    try:
        while time.perf_counter() - start_time < duration_seconds:
            acq_start = time.perf_counter()
            
            frame_result = frame_manager.get_latest_frame()
            
            acq_time = time.perf_counter() - acq_start
            acquisition_times.append(acq_time)
            
            if frame_result is None:
                time.sleep(0.005)
                continue
                
            fd, frame_timestamp = frame_result
            frame_count += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        frame_manager.stop()
    
    total_time = time.perf_counter() - start_time
    
    # Get frame manager stats
    stats = frame_manager.get_stats()
    
    return {
        'method': 'optimized',
        'duration': total_time,
        'frames_processed': frame_count,
        'fps': frame_count / total_time,
        'avg_acquisition_ms': statistics.mean(acquisition_times) * 1000,
        'max_acquisition_ms': max(acquisition_times) * 1000,
        'median_acquisition_ms': statistics.median(acquisition_times) * 1000,
        'drop_rate': stats.get('drop_rate', 0) if stats else 0,
        'frames_received': stats.get('frames_received', 0) if stats else 0
    }


def compare_methods(test_duration: int = 30):
    """Compare both acquisition methods."""
    print(f"Running performance comparison for {test_duration} seconds each...\n")
    
    # Test original method
    original_results = test_original_acquisition(test_duration)
    
    print("\nWaiting 5 seconds between tests...\n")
    time.sleep(5)
    
    # Test optimized method
    optimized_results = test_optimized_acquisition(test_duration)
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*60)
    
    print(f"{'Metric':<25} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 60)
    
    metrics = [
        ('FPS', 'fps', '{:.1f}'),
        ('Avg Acquisition (ms)', 'avg_acquisition_ms', '{:.2f}'),
        ('Max Acquisition (ms)', 'max_acquisition_ms', '{:.2f}'),
        ('Median Acquisition (ms)', 'median_acquisition_ms', '{:.2f}'),
        ('Frames Processed', 'frames_processed', '{:d}')
    ]
    
    for name, key, fmt in metrics:
        orig_val = original_results[key]
        opt_val = optimized_results[key]
        
        if key == 'fps' or key == 'frames_processed':
            improvement = f"{((opt_val / orig_val - 1) * 100):+.1f}%"
        else:
            improvement = f"{((orig_val / opt_val - 1) * 100):+.1f}%"
        
        print(f"{name:<25} {fmt.format(orig_val):<15} {fmt.format(opt_val):<15} {improvement:<15}")
    
    # Additional optimized-only metrics
    if 'drop_rate' in optimized_results:
        print(f"{'Drop Rate':<25} {'N/A':<15} {optimized_results['drop_rate']:.1%:<15} {'New Metric':<15}")
    
    print("-" * 60)
    
    # Summary
    fps_improvement = (optimized_results['fps'] / original_results['fps'] - 1) * 100
    latency_improvement = (original_results['avg_acquisition_ms'] / optimized_results['avg_acquisition_ms'] - 1) * 100
    
    print(f"\nSUMMARY:")
    print(f"  FPS Improvement: {fps_improvement:+.1f}%")
    print(f"  Latency Improvement: {latency_improvement:+.1f}%")
    
    if fps_improvement > 10:
        print("  ✅ Significant performance improvement achieved!")
    elif fps_improvement > 0:
        print("  ✓ Modest performance improvement")
    else:
        print("  ⚠️ Performance regression detected")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance test for frame acquisition')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--method', choices=['original', 'optimized', 'compare'], default='compare',
                       help='Which method to test')
    
    args = parser.parse_args()
    
    if args.method == 'original':
        results = test_original_acquisition(args.duration)
        print(f"Original method results: {results}")
    elif args.method == 'optimized':
        results = test_optimized_acquisition(args.duration)
        print(f"Optimized method results: {results}")
    else:
        compare_methods(args.duration) 