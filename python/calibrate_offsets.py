#!/usr/bin/env python3
"""
IMU calibration program for D435i camera.
Collects stationary samples to compute gyro and accelerometer offsets,
then demonstrates before/after correction.
"""

import time
import numpy as np
import sys
import os
from datetime import datetime

# Add the parent directory to sys.path to import the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import zenoh_d435i_subscriber as zd435i
except ImportError as e:
    print(f"Error importing zenoh_d435i_subscriber: {e}")
    print("Make sure the module is compiled with: maturin develop")
    sys.exit(1)

from imu_utils import (
    collect_calibration_samples, compute_calibration, save_calibration,
    load_calibration, validate_calibration_quality, print_quality_report,
    IMUProcessor
)


def show_before_after_comparison(subscriber, calibration, duration_seconds: float = 10.0):
    """Show real-time comparison of raw vs calibrated IMU data."""
    print(f"\n=== Before/After Comparison ({duration_seconds}s) ===")
    print("Move the device around to see the difference...")
    
    processor = IMUProcessor(calibration)
    start_time = time.time()
    last_frame_id = -1
    sample_count = 0
    
    while time.time() - start_time < duration_seconds:
        frame_data = subscriber.get_latest_frames()
        
        if (frame_data.frame_count == 0 or 
            frame_data.frame_count == last_frame_id or 
            frame_data.motion is None):
            time.sleep(0.01)
            continue
        
        last_frame_id = frame_data.frame_count
        sample_count += 1
        
        # Raw data
        raw_gyro = np.array(frame_data.motion.gyro)
        raw_accel = np.array(frame_data.motion.accel)
        
        # Apply calibration
        cal_gyro, cal_accel = processor.process_sample(raw_gyro, raw_accel)
        
        # Print comparison every 20 samples
        if sample_count % 20 == 0:
            print(f"\nSample {sample_count}:")
            print(f"  Raw Gyro:   [{raw_gyro[0]:+8.4f}, {raw_gyro[1]:+8.4f}, {raw_gyro[2]:+8.4f}] rad/s")
            print(f"  Cal Gyro:   [{cal_gyro[0]:+8.4f}, {cal_gyro[1]:+8.4f}, {cal_gyro[2]:+8.4f}] rad/s")
            print(f"  Raw Accel:  [{raw_accel[0]:+8.4f}, {raw_accel[1]:+8.4f}, {raw_accel[2]:+8.4f}] m/s²")
            print(f"  Cal Accel:  [{cal_accel[0]:+8.4f}, {cal_accel[1]:+8.4f}, {cal_accel[2]:+8.4f}] m/s²")
            
            # Show magnitudes
            raw_gyro_mag = np.linalg.norm(raw_gyro)
            cal_gyro_mag = np.linalg.norm(cal_gyro)
            raw_accel_mag = np.linalg.norm(raw_accel)
            cal_accel_mag = np.linalg.norm(cal_accel)
            
            print(f"  Gyro |mag|: {raw_gyro_mag:.4f} -> {cal_gyro_mag:.4f} rad/s")
            print(f"  Accel |mag|: {raw_accel_mag:.4f} -> {cal_accel_mag:.4f} m/s²")
    
    # Final statistics
    stats_raw = processor.get_statistics(use_calibrated=False)
    stats_cal = processor.get_statistics(use_calibrated=True)
    
    print(f"\n=== Final Statistics ({sample_count} samples) ===")
    
    if "error" not in stats_raw:
        print("Raw Data:")
        print(f"  Gyro mean:  [{stats_raw['gyro']['mean'][0]:+8.5f}, {stats_raw['gyro']['mean'][1]:+8.5f}, {stats_raw['gyro']['mean'][2]:+8.5f}] rad/s")
        print(f"  Gyro std:   [{stats_raw['gyro']['std'][0]:8.5f}, {stats_raw['gyro']['std'][1]:8.5f}, {stats_raw['gyro']['std'][2]:8.5f}] rad/s")
        print(f"  Accel mean: [{stats_raw['accel']['mean'][0]:+8.5f}, {stats_raw['accel']['mean'][1]:+8.5f}, {stats_raw['accel']['mean'][2]:+8.5f}] m/s²")
        print(f"  Accel std:  [{stats_raw['accel']['std'][0]:8.5f}, {stats_raw['accel']['std'][1]:8.5f}, {stats_raw['accel']['std'][2]:8.5f}] m/s²")
    
    if "error" not in stats_cal:
        print("Calibrated Data:")
        print(f"  Gyro mean:  [{stats_cal['gyro']['mean'][0]:+8.5f}, {stats_cal['gyro']['mean'][1]:+8.5f}, {stats_cal['gyro']['mean'][2]:+8.5f}] rad/s")
        print(f"  Gyro std:   [{stats_cal['gyro']['std'][0]:8.5f}, {stats_cal['gyro']['std'][1]:8.5f}, {stats_cal['gyro']['std'][2]:8.5f}] rad/s")
        print(f"  Accel mean: [{stats_cal['accel']['mean'][0]:+8.5f}, {stats_cal['accel']['mean'][1]:+8.5f}, {stats_cal['accel']['mean'][2]:+8.5f}] m/s²")
        print(f"  Accel std:  [{stats_cal['accel']['std'][0]:8.5f}, {stats_cal['accel']['std'][1]:8.5f}, {stats_cal['accel']['std'][2]:8.5f}] m/s²")


def main():
    """Main calibration workflow."""
    print("=== D435i IMU Calibration Program ===")
    print("This program will calibrate your IMU by removing biases and gravity.")
    print()
    
    # Check for existing calibration
    existing_cal = load_calibration()
    if existing_cal is not None:
        print(f"Found existing calibration from {datetime.fromtimestamp(existing_cal.timestamp)}")
        choice = input("Use existing calibration? (y/n/t for test): ").lower().strip()
        
        if choice == 'y':
            print("Using existing calibration for comparison demo.")
            use_existing = True
        elif choice == 't':
            print("Testing existing calibration...")
            use_existing = True
        else:
            print("Will create new calibration.")
            use_existing = False
    else:
        print("No existing calibration found. Will create new one.")
        use_existing = False
    
    # Set up subscriber
    subscriber = zd435i.ZenohD435iSubscriber()
    
    print("\nConnecting to Zenoh...")
    try:
        subscriber.connect()
        print("✓ Connected to Zenoh")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    print("Starting subscriptions...")
    try:
        subscriber.start_subscribing()
        print("✓ Subscriptions started")
    except Exception as e:
        print(f"✗ Failed to start subscriptions: {e}")
        return
    
    # Wait for initial data
    print("Waiting for IMU data...")
    while True:
        frame_data = subscriber.get_latest_frames()
        if frame_data.frame_count > 0 and frame_data.motion is not None:
            print("✓ IMU data received")
            break
        time.sleep(0.1)
    
    calibration = None
    
    try:
        if not use_existing:
            # Perform new calibration
            print(f"\n{'='*60}")
            print("CALIBRATION PHASE")
            print(f"{'='*60}")
            
            # Ask user to position device
            input("\nPlace the device on a stable, level surface and press Enter...")
            print("Starting calibration in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
            
            # Collect samples
            try:
                gyro_samples, accel_samples = collect_calibration_samples(
                    subscriber, num_samples=50, timeout_seconds=30.0
                )
            except TimeoutError as e:
                print(f"✗ Calibration failed: {e}")
                return
            
            # Validate sample quality
            quality = validate_calibration_quality(gyro_samples, accel_samples)
            print_quality_report(quality)
            
            if quality['overall_quality'] == 'poor':
                choice = input("\nCalibration quality is poor. Continue anyway? (y/n): ").lower().strip()
                if choice != 'y':
                    print("Calibration aborted.")
                    return
            
            # Compute calibration
            calibration = compute_calibration(gyro_samples, accel_samples)
            
            # Save calibration
            save_calibration(calibration)
            
        else:
            calibration = existing_cal
        
        # Demonstration phase
        if calibration is not None:
            print(f"\n{'='*60}")
            print("DEMONSTRATION PHASE")
            print(f"{'='*60}")
            
            choice = input("\nShow before/after comparison? (y/n): ").lower().strip()
            if choice == 'y':
                show_before_after_comparison(subscriber, calibration, duration_seconds=15.0)
            
            # Test with processor
            print(f"\n{'='*60}")
            print("REAL-TIME CALIBRATED DATA (10 seconds)")
            print(f"{'='*60}")
            print("This shows live calibrated IMU data...")
            
            processor = IMUProcessor(calibration)
            start_time = time.time()
            last_frame_id = -1
            sample_count = 0
            
            while time.time() - start_time < 10.0:
                frame_data = subscriber.get_latest_frames()
                
                if (frame_data.frame_count == 0 or 
                    frame_data.frame_count == last_frame_id or 
                    frame_data.motion is None):
                    time.sleep(0.01)
                    continue
                
                last_frame_id = frame_data.frame_count
                sample_count += 1
                
                # Process with calibration
                raw_gyro = np.array(frame_data.motion.gyro)
                raw_accel = np.array(frame_data.motion.accel)
                cal_gyro, cal_accel = processor.process_sample(raw_gyro, raw_accel)
                
                # Print every 30 samples
                if sample_count % 30 == 0:
                    print(f"[{sample_count:3d}] Gyro: [{cal_gyro[0]:+7.4f}, {cal_gyro[1]:+7.4f}, {cal_gyro[2]:+7.4f}] | "
                          f"Accel: [{cal_accel[0]:+7.4f}, {cal_accel[1]:+7.4f}, {cal_accel[2]:+7.4f}]")
            
            print(f"\n✓ Processed {sample_count} samples")
            
            # Final summary
            print(f"\n{'='*60}")
            print("CALIBRATION SUMMARY")
            print(f"{'='*60}")
            print(f"Calibration file: imu_calibration.json")
            print(f"Created: {datetime.fromtimestamp(calibration.timestamp)}")
            print(f"Samples used: {calibration.num_samples}")
            print(f"Gyro offset:  [{calibration.gyro_offset[0]:+8.5f}, {calibration.gyro_offset[1]:+8.5f}, {calibration.gyro_offset[2]:+8.5f}] rad/s")
            print(f"Accel offset: [{calibration.accel_offset[0]:+8.5f}, {calibration.accel_offset[1]:+8.5f}, {calibration.accel_offset[2]:+8.5f}] m/s²")
            print(f"Gravity:      [{calibration.gravity_vector[0]:+8.5f}, {calibration.gravity_vector[1]:+8.5f}, {calibration.gravity_vector[2]:+8.5f}] m/s²")
            print()
            print("You can now use this calibration in other programs by:")
            print("  from imu_utils import load_calibration, IMUProcessor")
            print("  calibration = load_calibration()")
            print("  processor = IMUProcessor(calibration)")
            print("  cal_gyro, cal_accel = processor.apply_calibration(raw_gyro, raw_accel)")
        
    except KeyboardInterrupt:
        print(f"\n\nCalibration interrupted by user.")
        
    finally:
        print("\nShutting down...")
        subscriber.stop()
        print("Done!")


if __name__ == "__main__":
    main() 