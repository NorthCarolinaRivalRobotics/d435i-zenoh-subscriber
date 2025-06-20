#!/usr/bin/env python3
"""
Coordinate frame testing script for IMU data.
This script helps identify the correct coordinate transformations needed
to align gyroscope and accelerometer frames.
"""

import time
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import zenoh_d435i_subscriber as zd435i
except ImportError as e:
    print(f"Error importing zenoh_d435i_subscriber: {e}")
    print("Make sure the module is compiled with: maturin develop")
    sys.exit(1)

from imu_utils import (
    load_calibration, IMUProcessor, OrientationEstimator,
    create_axis_swap_matrix, create_common_transforms
)


def test_coordinate_transforms():
    """Test different coordinate frame transformations to find the correct alignment."""
    print("=== IMU Coordinate Frame Testing ===")
    print("This script will help identify coordinate frame issues between gyro and accel.")
    print("Move the device around and observe which transforms fix the alignment.\n")
    
    # Load calibration
    calibration = load_calibration()
    if calibration is None:
        print("Warning: No calibration found. Using uncalibrated data.")
    
    # Create available transforms
    transforms = create_common_transforms()
    
    # Add some specific D435i-likely transforms based on your observations
    transforms.update({
        'accel_fix_x': create_axis_swap_matrix('-x', 'y', 'z'),  # Negate X for accel
        'accel_swap_yz': create_axis_swap_matrix('x', 'z', 'y'),  # Swap Y/Z for accel
        'accel_fix_x_swap_yz': create_axis_swap_matrix('-x', 'z', 'y'),  # Both fixes
        'gyro_fix_x': create_axis_swap_matrix('-x', 'y', 'z'),   # Negate X for gyro
        'gyro_swap_yz': create_axis_swap_matrix('x', 'z', 'y'),   # Swap Y/Z for gyro
        'gyro_fix_x_swap_yz': create_axis_swap_matrix('-x', 'z', 'y'),  # Both fixes
    })
    
    # Set up subscriber and processor
    subscriber = zd435i.ZenohD435iSubscriber()
    processor = IMUProcessor(calibration)
    
    print("Available transforms:")
    for name, transform in transforms.items():
        print(f"  {name}: \n{transform}")
        print()
    
    # Menu for selecting transforms
    print("Select transforms to test:")
    print("1. Test accelerometer transforms (gyro unchanged)")
    print("2. Test gyroscope transforms (accel unchanged)")
    print("3. Test specific combination")
    print("4. Interactive mode - try different combinations")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        test_accel_transforms(subscriber, processor, transforms)
    elif choice == "2":
        test_gyro_transforms(subscriber, processor, transforms)
    elif choice == "3":
        test_specific_combination(subscriber, processor, transforms)
    elif choice == "4":
        interactive_testing(subscriber, processor, transforms)
    else:
        print("Invalid choice")


def test_accel_transforms(subscriber, processor, transforms):
    """Test different accelerometer transforms."""
    print("\n=== Testing Accelerometer Transforms ===")
    print("Gyroscope will remain unchanged. Only accelerometer transforms will be tested.")
    print("Tilt the device around different axes and observe which transform makes")
    print("the accelerometer-only estimate match the gyro-only estimate better.\n")
    
    # Connect and start
    subscriber.connect()
    subscriber.start_subscribing()
    
    # Wait for data
    while True:
        frame_data = subscriber.get_latest_frames()
        if frame_data.frame_count > 0 and frame_data.motion is not None:
            break
        time.sleep(0.1)
    
    accel_transform_names = ['identity', 'accel_fix_x', 'accel_swap_yz', 'accel_fix_x_swap_yz']
    
    for transform_name in accel_transform_names:
        print(f"\n--- Testing accelerometer transform: {transform_name} ---")
        accel_transform = transforms[transform_name]
        
        # Create estimator with this transform
        estimator = OrientationEstimator(
            accel_frame_transform=accel_transform,
            gyro_frame_transform=np.eye(3)  # Identity for gyro
        )
        
        print("Starting data collection... (Press Enter to try next transform)")
        
        last_frame_id = -1
        last_timestamp = None
        frame_count = 0
        
        while True:
            # Check for keyboard input (non-blocking)
            import select
            import sys
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                input()  # Consume the input
                break
            
            frame_data = subscriber.get_latest_frames()
            
            if frame_data.frame_count == 0 or frame_data.frame_count == last_frame_id:
                time.sleep(0.01)
                continue
                
            if frame_data.motion is None:
                continue
                
            last_frame_id = frame_data.frame_count
            frame_count += 1
            
            # Process data
            raw_gyro = np.array(frame_data.motion.gyro)
            raw_accel = np.array(frame_data.motion.accel)
            timestamp = frame_data.motion.timestamp
            
            cal_gyro, cal_accel = processor.process_sample(raw_gyro, raw_accel)
            
            if last_timestamp is not None:
                dt = (timestamp - last_timestamp) / 1000.0
            else:
                dt = 0.01
            last_timestamp = timestamp
            
            # Update estimator
            fused_orientation, gyro_bias = estimator.update(cal_gyro, cal_accel, dt)
            gyro_only, accel_only, fused = estimator.get_all_orientations()
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                gyro_euler = np.degrees(gyro_only.to_euler_angles())
                accel_euler = np.degrees(accel_only.to_euler_angles())
                fused_euler = np.degrees(fused.to_euler_angles())
                
                print(f"Frame {frame_count}:")
                print(f"  Raw Accel: [{raw_accel[0]:+7.3f}, {raw_accel[1]:+7.3f}, {raw_accel[2]:+7.3f}] m/s²")
                print(f"  Transformed: [{(accel_transform @ cal_accel)[0]:+7.3f}, {(accel_transform @ cal_accel)[1]:+7.3f}, {(accel_transform @ cal_accel)[2]:+7.3f}] m/s²")
                print(f"  Gyro-only (RPY):  [{gyro_euler[0]:+7.2f}, {gyro_euler[1]:+7.2f}, {gyro_euler[2]:+7.2f}]")
                print(f"  Accel-only (RPY): [{accel_euler[0]:+7.2f}, {accel_euler[1]:+7.2f}, {accel_euler[2]:+7.2f}]")
                print(f"  Fused (RPY):      [{fused_euler[0]:+7.2f}, {fused_euler[1]:+7.2f}, {fused_euler[2]:+7.2f}]")
                
                # Check alignment
                angle_diff = np.abs(gyro_euler - accel_euler)
                avg_diff = np.mean(angle_diff[:2])  # Roll and pitch only (yaw undefined for accel)
                print(f"  Avg RP difference: {avg_diff:.2f}°")
                
                # Debug stats
                debug = estimator.get_debug_stats()
                print(f"  Gravity direction: [{debug['avg_gravity_direction'][0]:+6.3f}, {debug['avg_gravity_direction'][1]:+6.3f}, {debug['avg_gravity_direction'][2]:+6.3f}]")
                print(f"  Accel usage: {debug['accel_usage_rate']:.1f}%")
                print()
        
        print(f"Finished testing {transform_name}")
    
    subscriber.stop()


def test_gyro_transforms(subscriber, processor, transforms):
    """Test different gyroscope transforms."""
    print("\n=== Testing Gyroscope Transforms ===")
    print("Similar to accel testing but for gyroscope...")
    # Similar implementation to test_accel_transforms but for gyro
    pass


def test_specific_combination(subscriber, processor, transforms):
    """Test a specific combination of transforms."""
    print("\n=== Testing Specific Combination ===")
    
    print("Available transforms:")
    for i, name in enumerate(transforms.keys()):
        print(f"  {i}: {name}")
    
    print("\nSelect accelerometer transform:")
    accel_idx = int(input("Enter number: "))
    accel_name = list(transforms.keys())[accel_idx]
    
    print("\nSelect gyroscope transform:")
    gyro_idx = int(input("Enter number: "))
    gyro_name = list(transforms.keys())[gyro_idx]
    
    print(f"\nTesting combination: accel={accel_name}, gyro={gyro_name}")
    
    # Run test with this combination
    run_combination_test(subscriber, processor, 
                        transforms[accel_name], transforms[gyro_name],
                        accel_name, gyro_name)


def run_combination_test(subscriber, processor, accel_transform, gyro_transform, 
                        accel_name, gyro_name):
    """Run a test with specific transform combination."""
    subscriber.connect()
    subscriber.start_subscribing()
    
    # Wait for data
    while True:
        frame_data = subscriber.get_latest_frames()
        if frame_data.frame_count > 0 and frame_data.motion is not None:
            break
        time.sleep(0.1)
    
    estimator = OrientationEstimator(
        accel_frame_transform=accel_transform,
        gyro_frame_transform=gyro_transform
    )
    
    print(f"Testing: accel={accel_name}, gyro={gyro_name}")
    print("Accel transform matrix:")
    print(accel_transform)
    print("Gyro transform matrix:")
    print(gyro_transform)
    print("\nMove the device and observe orientation estimates...")
    print("Press Ctrl+C to stop\n")
    
    last_frame_id = -1
    last_timestamp = None
    frame_count = 0
    
    try:
        while True:
            frame_data = subscriber.get_latest_frames()
            
            if frame_data.frame_count == 0 or frame_data.frame_count == last_frame_id:
                time.sleep(0.01)
                continue
                
            if frame_data.motion is None:
                continue
                
            last_frame_id = frame_data.frame_count
            frame_count += 1
            
            # Process data
            raw_gyro = np.array(frame_data.motion.gyro)
            raw_accel = np.array(frame_data.motion.accel)
            timestamp = frame_data.motion.timestamp
            
            cal_gyro, cal_accel = processor.process_sample(raw_gyro, raw_accel)
            
            if last_timestamp is not None:
                dt = (timestamp - last_timestamp) / 1000.0
            else:
                dt = 0.01
            last_timestamp = timestamp
            
            # Update estimator
            fused_orientation, gyro_bias = estimator.update(cal_gyro, cal_accel, dt)
            gyro_only, accel_only, fused = estimator.get_all_orientations()
            
            # Print status every 25 frames
            if frame_count % 25 == 0:
                gyro_euler = np.degrees(gyro_only.to_euler_angles())
                accel_euler = np.degrees(accel_only.to_euler_angles())
                fused_euler = np.degrees(fused.to_euler_angles())
                
                print(f"--- Frame {frame_count} ---")
                print(f"Raw Data:")
                print(f"  Gyro:  [{raw_gyro[0]:+7.4f}, {raw_gyro[1]:+7.4f}, {raw_gyro[2]:+7.4f}] rad/s")
                print(f"  Accel: [{raw_accel[0]:+7.3f}, {raw_accel[1]:+7.3f}, {raw_accel[2]:+7.3f}] m/s²")
                
                print(f"Transformed Data:")
                gyro_transformed = gyro_transform @ cal_gyro
                accel_transformed = accel_transform @ cal_accel
                print(f"  Gyro:  [{gyro_transformed[0]:+7.4f}, {gyro_transformed[1]:+7.4f}, {gyro_transformed[2]:+7.4f}] rad/s")
                print(f"  Accel: [{accel_transformed[0]:+7.3f}, {accel_transformed[1]:+7.3f}, {accel_transformed[2]:+7.3f}] m/s²")
                
                print(f"Orientation Estimates (Roll, Pitch, Yaw):")
                print(f"  Gyro-only:  [{gyro_euler[0]:+7.2f}, {gyro_euler[1]:+7.2f}, {gyro_euler[2]:+7.2f}]°")
                print(f"  Accel-only: [{accel_euler[0]:+7.2f}, {accel_euler[1]:+7.2f}, {accel_euler[2]:+7.2f}]°")
                print(f"  Fused:      [{fused_euler[0]:+7.2f}, {fused_euler[1]:+7.2f}, {fused_euler[2]:+7.2f}]°")
                
                # Analyze alignment
                roll_diff = abs(gyro_euler[0] - accel_euler[0])
                pitch_diff = abs(gyro_euler[1] - accel_euler[1])
                print(f"  Differences: Roll={roll_diff:.1f}°, Pitch={pitch_diff:.1f}°")
                
                # Debug info
                debug = estimator.get_debug_stats()
                print(f"  Gravity dir: [{debug['avg_gravity_direction'][0]:+6.3f}, {debug['avg_gravity_direction'][1]:+6.3f}, {debug['avg_gravity_direction'][2]:+6.3f}]")
                print(f"  Accel usage: {debug['accel_usage_rate']:.1f}%")
                print()
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    
    finally:
        subscriber.stop()


def interactive_testing(subscriber, processor, transforms):
    """Interactive mode for testing different combinations."""
    print("\n=== Interactive Testing Mode ===")
    print("Type transform combinations to test them in real-time.")
    print("Format: accel_transform,gyro_transform")
    print("Example: accel_fix_x,identity")
    print("Type 'list' to see available transforms")
    print("Type 'quit' to exit")
    
    while True:
        command = input("\nEnter command: ").strip()
        
        if command == 'quit':
            break
        elif command == 'list':
            print("Available transforms:")
            for name in transforms.keys():
                print(f"  {name}")
            continue
        
        try:
            accel_name, gyro_name = command.split(',')
            accel_name = accel_name.strip()
            gyro_name = gyro_name.strip()
            
            if accel_name not in transforms or gyro_name not in transforms:
                print("Unknown transform name. Type 'list' to see available transforms.")
                continue
            
            print(f"Testing: accel={accel_name}, gyro={gyro_name}")
            run_combination_test(subscriber, processor,
                               transforms[accel_name], transforms[gyro_name],
                               accel_name, gyro_name)
            
        except ValueError:
            print("Invalid format. Use: accel_transform,gyro_transform")
        except KeyboardInterrupt:
            print("\nTest interrupted.")


if __name__ == "__main__":
    test_coordinate_transforms() 