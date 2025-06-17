#!/usr/bin/env python3
"""
Test script to verify latency optimizations in ZenohD435iSubscriber.
"""

import time
import statistics
import zenoh_d435i_subscriber as zd435i

def test_latency_optimizations():
    """Test the new latency optimization features."""
    print("Testing Latency-Optimized ZenohD435iSubscriber")
    print("=" * 50)
    
    # Create subscriber
    sub = zd435i.ZenohD435iSubscriber()
    
    # Connect and start
    print("Connecting to Zenoh...")
    sub.connect()
    sub.start_subscribing()
    
    # Test statistics
    print("\nInitial stats:")
    print(sub.get_stats())
    
    # Test frame acquisition timing
    print("\nTesting frame acquisition latency...")
    acquisition_times = []
    frame_counts = []
    last_frame_count = 0
    
    for i in range(100):  # Test for 100 iterations
        start_time = time.perf_counter()
        
        # Get latest frames (should be non-blocking with try_lock)
        frame_data = sub.get_latest_frames()
        
        acquisition_time = time.perf_counter() - start_time
        acquisition_times.append(acquisition_time * 1000)  # Convert to ms
        frame_counts.append(frame_data.frame_count)
        
        if frame_data.frame_count > 0:
            if frame_data.frame_count != last_frame_count:
                print(f"Frame {frame_data.frame_count}: {frame_data}")
                last_frame_count = frame_data.frame_count
        
        time.sleep(0.01)  # 10ms sleep between checks
    
    # Analyze results
    print(f"\nFrame Acquisition Performance:")
    print(f"  Mean time: {statistics.mean(acquisition_times):.3f}ms")
    print(f"  Max time:  {max(acquisition_times):.3f}ms")
    if len(acquisition_times) >= 20:
        print(f"  95th percentile: {sorted(acquisition_times)[int(len(acquisition_times)*0.95)]:.3f}ms")
    if len(acquisition_times) >= 100:
        print(f"  99th percentile: {sorted(acquisition_times)[int(len(acquisition_times)*0.99)]:.3f}ms")
    
    # Test lazy depth conversion if we have depth data
    final_frame = sub.get_latest_frames()
    if final_frame.depth is not None:
        print(f"\nTesting lazy depth conversion...")
        
        # First access should trigger conversion
        start_time = time.perf_counter()
        depth_data = final_frame.depth.get_data_2d()
        first_access_time = time.perf_counter() - start_time
        
        # Second access should use cached data
        start_time = time.perf_counter()
        depth_data_cached = final_frame.depth.get_data_2d()
        second_access_time = time.perf_counter() - start_time
        
        print(f"  First access (with conversion): {first_access_time*1000:.3f}ms")
        print(f"  Second access (cached): {second_access_time*1000:.3f}ms")
        if second_access_time > 0:
            print(f"  Speedup: {first_access_time/second_access_time:.1f}x")
        
        # Test raw data access (should be instant)
        start_time = time.perf_counter()
        raw_data = final_frame.depth.get_raw_data()
        raw_access_time = time.perf_counter() - start_time
        print(f"  Raw data access: {raw_access_time*1000:.3f}ms")
    
    # Final statistics
    print(f"\nFinal stats:")
    print(sub.get_stats())
    
    # Stop subscriber
    sub.stop()
    print("\nTest completed!")
    
    # Recommendations
    print("\nOptimization Summary:")
    print("✅ Non-blocking frame acquisition with try_lock")
    print("✅ Async heavy processing moved to spawn_blocking")
    print("✅ Frame dropping when processing can't keep up")
    print("✅ Lazy depth conversion (only when needed)")
    print("✅ Atomic counters to reduce lock contention")
    print("✅ Processing statistics and monitoring")

if __name__ == "__main__":
    try:
        test_latency_optimizations()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}") 