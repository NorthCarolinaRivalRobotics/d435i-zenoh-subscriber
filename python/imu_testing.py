#!/usr/bin/env python3
"""
Advanced IMU testing program with Lie group-based orientation estimation.
Uses proper SO(3) manifold integration with complementary filtering and 
Rerun visualization showing three orientation estimates:
1. Gyro-only integration (manifold-aware)
2. Accelerometer-only orientation (gravity direction)
3. Fused complementary filter result

Coordinate transforms are hardcoded based on testing:
- Gyroscope: swap Y and Z axes
- Accelerometer: negate X and Y axes
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

# Import Rerun for visualization
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    print("Warning: Rerun not available. Install with: pip install rerun-sdk")
    HAS_RERUN = False

from imu_utils import (
    MahonyFilter, load_calibration, IMUProcessor,
    create_axis_swap_matrix
)


class RerunOrientationVisualizer:
    """Rerun-based 3D orientation visualizer."""
    
    def __init__(self, app_id: str = "imu_orientation_test"):
        """Initialize Rerun logging."""
        if not HAS_RERUN:
            print("Rerun not available - visualization disabled")
            return
            
        rr.init(app_id, spawn=True)
        
        # Set up the 3D view
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Create coordinate frame
        self.setup_world_frame()
        
        # Cube size
        self.cube_size = 0.2
        
    def setup_world_frame(self):
        """Set up world coordinate frame."""
        if not HAS_RERUN:
            return
            
        # World coordinate axes
        rr.log("world/axes", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"]
        ), static=True)
        
        # Ground plane
        rr.log("world/ground", rr.Mesh3D(
            vertex_positions=[[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
            triangle_indices=[[0, 1, 2], [0, 2, 3]],
            albedo_factor=[0.5, 0.5, 0.5, 0.3]
        ), static=True)
    
    def log_orientation_cubes(self, gyro_quat, accel_quat, fused_quat):
        """Log three cubes showing different orientation estimates."""
        if not HAS_RERUN:
            return
        
        # Cube positions (spread them out)
        positions = [
            [-1.1, 0.0, 0.5],  # Gyro-only (left)
            [0.0, 0.0, 0.5],   # Accelerometer-only (center)
            [1.1, 0.0, 0.5]    # Fused (right)
        ]
        
        # Colors for each cube
        colors = [
            [255, 100, 100, 200],  # Red for gyro-only
            [100, 255, 100, 200],  # Green for accel-only
            [100, 100, 255, 200]   # Blue for fused
        ]
        
        # Labels
        labels = ["Gyro Only", "Accel Only", "Fused"]
        orientations = [gyro_quat, accel_quat, fused_quat]
        
        for i, (pos, color, label, quat) in enumerate(zip(positions, colors, labels, orientations)):
            # Create cube vertices (unit cube centered at origin)
            s = self.cube_size / 2
            vertices = np.array([
                [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom
                [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]       # Top
            ])
            
            # Rotate vertices by quaternion
            rotated_vertices = np.array([quat.rotate_vector(v) for v in vertices])
            
            # Translate to position
            final_vertices = rotated_vertices + np.array(pos)
            
            # Define cube faces
            triangles = [
                [0, 1, 2], [0, 2, 3],  # Bottom
                [4, 7, 6], [4, 6, 5],  # Top
                [0, 4, 5], [0, 5, 1],  # Front
                [2, 6, 7], [2, 7, 3],  # Back
                [0, 3, 7], [0, 7, 4],  # Left
                [1, 5, 6], [1, 6, 2]   # Right
            ]
            
            # Log the cube
            rr.log(f"cubes/{label.lower().replace(' ', '_')}", rr.Mesh3D(
                vertex_positions=final_vertices,
                triangle_indices=triangles,
                albedo_factor=color
            ))
            
            # Log text label using Points3D with labels instead of Text3D
            rr.log(f"cubes/{label.lower().replace(' ', '_')}/label", rr.Points3D(
                positions=[[pos[0], pos[1], pos[2] + 0.4]],
                colors=[color[:3]],
                labels=[label],
                radii=[0.05]
            ))
            
            # Log orientation as arrow pointing forward
            forward = quat.rotate_vector(np.array([0.3, 0, 0]))  # Forward direction
            rr.log(f"cubes/{label.lower().replace(' ', '_')}/forward", rr.Arrows3D(
                origins=[pos],
                vectors=[forward],
                colors=[color[:3]]
            ))
    
    def log_imu_data(self, gyro, accel, gyro_bias, timestamp):
        """Log raw IMU data for analysis."""
        if not HAS_RERUN:
            return
        
        # Log gyro data
        rr.set_time_seconds("timestamp", timestamp)
        rr.log("imu/gyro_magnitude", rr.Scalar(np.linalg.norm(gyro)))
        rr.log("imu/gyro_x", rr.Scalar(gyro[0]))
        rr.log("imu/gyro_y", rr.Scalar(gyro[1]))
        rr.log("imu/gyro_z", rr.Scalar(gyro[2]))
        
        # Log accel data
        rr.log("imu/accel_magnitude", rr.Scalar(np.linalg.norm(accel)))
        rr.log("imu/accel_x", rr.Scalar(accel[0]))
        rr.log("imu/accel_y", rr.Scalar(accel[1]))
        rr.log("imu/accel_z", rr.Scalar(accel[2]))
        
        # Log bias estimate
        rr.log("imu/bias_x", rr.Scalar(gyro_bias[0]))
        rr.log("imu/bias_y", rr.Scalar(gyro_bias[1]))
        rr.log("imu/bias_z", rr.Scalar(gyro_bias[2]))
    
    def log_euler_angles(self, gyro_quat, accel_quat, fused_quat, timestamp):
        """Log Euler angles for all three estimates."""
        if not HAS_RERUN:
            return
        
        rr.set_time_seconds("timestamp", timestamp)
        
        # Convert to Euler angles (degrees)
        for name, quat in [("gyro", gyro_quat), ("accel", accel_quat), ("fused", fused_quat)]:
            roll, pitch, yaw = quat.to_euler_angles()
            roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
            
            rr.log(f"euler/{name}/roll", rr.Scalar(roll_deg))
            rr.log(f"euler/{name}/pitch", rr.Scalar(pitch_deg))
            rr.log(f"euler/{name}/yaw", rr.Scalar(yaw_deg))


def main():
    """Main testing loop with orientation estimation and visualization."""
    print("=== Advanced D435i IMU Orientation Test ===")
    print("This program demonstrates proper Lie group integration and complementary filtering.")
    print("Using optimized coordinate transforms based on testing:")
    print("- Gyroscope: swap Y and Z axes")
    print("- Accelerometer: negate X and Y axes")
    print()
    
    # Define the optimal transforms found through testing
    gyro_transform = create_axis_swap_matrix('x', 'z', 'y')  # Swap Y and Z
    accel_transform = create_axis_swap_matrix('-x', '-y', 'z')  # Negate X and Y
    
    print("Applied coordinate transforms:")
    print("Gyroscope transform (swap Y/Z):")
    print(gyro_transform)
    print("Accelerometer transform (negate X/Y):")
    print(accel_transform)
    print()
    
    # Load calibration
    calibration = load_calibration()
    if calibration is None:
        print("Warning: No calibration found. Run calibrate_offsets.py first for best results.")
        print("Continuing with uncalibrated data...\n")
    else:
        print("✓ Using calibrated IMU data\n")
    
    # Set up components
    subscriber = zd435i.ZenohD435iSubscriber()
    processor = IMUProcessor(calibration)
    estimator = MahonyFilter(
        k_p=0.5, k_i=0.0, gravity_mag=9.81,  # Disabled bias estimation: k_i=0
        accel_frame_transform=accel_transform,
        gyro_frame_transform=gyro_transform
    )
    
    # Initialize visualization
    if HAS_RERUN:
        visualizer = RerunOrientationVisualizer()
        print("✓ Rerun visualization initialized")
    else:
        visualizer = None
        print("⚠ Rerun visualization not available")
    
    print("Connecting to Zenoh...")
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
        print("Waiting for data... (Press Ctrl+C to stop)")
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
    
    # Main processing loop
    frame_count = 0
    last_frame_id = -1
    last_stats_time = time.time()
    last_timestamp = None
    
    # Performance tracking
    orientation_times = []
    
    # Add reset capability
    reset_count = 0
    
    try:
        while True:
            # Get latest frame data
            frame_data = subscriber.get_latest_frames()
            
            # Check if we have new data
            if frame_data.frame_count == 0:
                time.sleep(0.001)  # 1ms wait
                continue
                
            if frame_data.frame_count == last_frame_id:
                time.sleep(0.001)
                continue
            
            if frame_data.motion is None:
                time.sleep(0.001)
                continue
            
            last_frame_id = frame_data.frame_count
            frame_count += 1
            
            # Extract and calibrate IMU data
            raw_gyro = np.array(frame_data.motion.gyro)
            raw_accel = np.array(frame_data.motion.accel)
            timestamp = frame_data.motion.timestamp
            
            cal_gyro, cal_accel = processor.process_sample(raw_gyro, raw_accel)
            if last_timestamp is not None:
                dt = (timestamp - last_timestamp) / 1000.0  # Convert ms to seconds
            else:
                print("No last timestamp")
                dt = 0.01  # Initial estimate: 10ms
            last_timestamp = timestamp
            # Check for excessive bias - reset if needed
            bias_magnitude = np.linalg.norm(estimator.gyro_bias)
            if bias_magnitude > 0.08 and frame_count % 100 == 0:  # Check every 100 frames
                print(f"WARNING: High bias magnitude {bias_magnitude:.5f} rad/s - consider reset")
                if reset_count < 3:  # Auto-reset up to 3 times
                    print("Auto-resetting orientation estimator...")
                    estimator.reset()
                    reset_count += 1
            
            # Orientation estimation
            orientation_start = time.perf_counter()
            
            # Update estimator
            fused_orientation, gyro_bias = estimator.update(cal_gyro, cal_accel, dt)
            
            # Get all three estimates
            gyro_only, accel_only, fused = estimator.get_all_orientations()
            
            orientation_time = time.perf_counter() - orientation_start
            orientation_times.append(orientation_time)
            
            # Visualization
            if visualizer is not None:
                visualizer.log_orientation_cubes(gyro_only, accel_only, fused)
                visualizer.log_imu_data(cal_gyro, cal_accel, gyro_bias, timestamp)
                visualizer.log_euler_angles(gyro_only, accel_only, fused, timestamp)
            
            # Print periodic status
            if frame_count % 50 == 0:  # Every 50 frames
                # Convert to Euler angles for display
                gyro_euler = np.degrees(gyro_only.to_euler_angles())
                accel_euler = np.degrees(accel_only.to_euler_angles())
                fused_euler = np.degrees(fused.to_euler_angles())
                
                print(f"\n--- Frame {frame_count} (Reset count: {reset_count}) ---")
                print(f"Timestamp: {timestamp:.3f}, dt: {dt:.4f}s")
                print(f"Raw Data:")
                print(f"  Gyro:  [{raw_gyro[0]:+7.4f}, {raw_gyro[1]:+7.4f}, {raw_gyro[2]:+7.4f}] rad/s")
                print(f"  Accel: [{raw_accel[0]:+7.3f}, {raw_accel[1]:+7.3f}, {raw_accel[2]:+7.3f}] m/s²")
                
                # Show transformed data
                gyro_transformed, accel_transformed = estimator.apply_coordinate_transforms(cal_gyro, cal_accel)
                print(f"Transformed Data:")
                print(f"  Gyro:  [{gyro_transformed[0]:+7.4f}, {gyro_transformed[1]:+7.4f}, {gyro_transformed[2]:+7.4f}] rad/s")
                print(f"  Accel: [{accel_transformed[0]:+7.3f}, {accel_transformed[1]:+7.3f}, {accel_transformed[2]:+7.3f}] m/s²")
                
                print(f"Calibrated (Pre-transform):")
                print(f"  Gyro:  [{cal_gyro[0]:+7.4f}, {cal_gyro[1]:+7.4f}, {cal_gyro[2]:+7.4f}] rad/s")
                print(f"  Accel: [{cal_accel[0]:+7.3f}, {cal_accel[1]:+7.3f}, {cal_accel[2]:+7.3f}] m/s²")
                print(f"Accel magnitude: {np.linalg.norm(cal_accel):.3f} m/s² (expect ~{estimator.gravity_mag:.1f})")
                print(f"Gyro bias: [{gyro_bias[0]:+6.4f}, {gyro_bias[1]:+6.4f}, {gyro_bias[2]:+6.4f}] rad/s (mag: {bias_magnitude:.5f})")
                
                # Debug stats
                debug_stats = estimator.get_debug_stats()
                print(f"Accel usage: {debug_stats['accel_usage_rate']:.1f}% ({debug_stats['accel_used']}/{debug_stats['total_updates']})")
                print(f"Gravity direction estimate: [{debug_stats['avg_gravity_direction'][0]:+6.3f}, {debug_stats['avg_gravity_direction'][1]:+6.3f}, {debug_stats['avg_gravity_direction'][2]:+6.3f}]")
                
                print(f"\nOrientation estimates (Roll, Pitch, Yaw in degrees):")
                print(f"  Gyro only:  [{gyro_euler[0]:+7.2f}, {gyro_euler[1]:+7.2f}, {gyro_euler[2]:+7.2f}]")
                print(f"  Accel only: [{accel_euler[0]:+7.2f}, {accel_euler[1]:+7.2f}, {accel_euler[2]:+7.2f}]")
                print(f"  Fused:      [{fused_euler[0]:+7.2f}, {fused_euler[1]:+7.2f}, {fused_euler[2]:+7.2f}]")
                
                # Coordinate frame analysis
                roll_diff = abs(gyro_euler[0] - accel_euler[0])
                pitch_diff = abs(gyro_euler[1] - accel_euler[1])
                print(f"  Roll/Pitch differences: {roll_diff:.1f}° / {pitch_diff:.1f}°")
                if roll_diff > 30 or pitch_diff > 30:
                    print(f"  ⚠ Large orientation differences - coordinate frame issue likely!")
                elif roll_diff < 10 and pitch_diff < 10:
                    print(f"  ✓ Good orientation alignment")
                else:
                    print(f"  ⚠ Moderate orientation differences")
            
            # Print statistics every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:
                elapsed = current_time - last_stats_time
                fps = frame_count / elapsed
                
                if orientation_times:
                    avg_orientation_time = sum(orientation_times) / len(orientation_times) * 1000
                    max_orientation_time = max(orientation_times) * 1000
                
                print(f"\n=== STATISTICS (last {elapsed:.1f}s) ===")
                print(f"Frame rate: {fps:.2f} FPS")
                print(f"Orientation processing: {avg_orientation_time:.2f}ms avg, {max_orientation_time:.2f}ms max")
                print(subscriber.get_stats())
                
                # Enhanced debug stats
                debug_stats = estimator.get_debug_stats()
                print(f"\nOrientation Filter Debug:")
                print(f"  Accelerometer usage: {debug_stats['accel_usage_rate']:.1f}% ({debug_stats['accel_used']}/{debug_stats['total_updates']} samples)")
                print(f"  Rejection rate: {debug_stats['accel_rejection_rate']:.1f}%")
                print(f"  Current bias magnitude: {debug_stats['bias_magnitude']:.5f} rad/s")
                print(f"  Bias vector: [{debug_stats['current_bias'][0]:+8.5f}, {debug_stats['current_bias'][1]:+8.5f}, {debug_stats['current_bias'][2]:+8.5f}]")
                
                # IMU statistics
                stats = processor.get_statistics(use_calibrated=True)
                if "error" not in stats:
                    print(f"\nCalibrated IMU stats ({stats['num_samples']} samples):")
                    print(f"  Gyro std:  [{stats['gyro']['std'][0]:.6f}, {stats['gyro']['std'][1]:.6f}, {stats['gyro']['std'][2]:.6f}] rad/s")
                    print(f"  Accel std: [{stats['accel']['std'][0]:.6f}, {stats['accel']['std'][1]:.6f}, {stats['accel']['std'][2]:.6f}] m/s²")
                    print(f"  Accel mean: [{stats['accel']['mean'][0]:+7.3f}, {stats['accel']['mean'][1]:+7.3f}, {stats['accel']['mean'][2]:+7.3f}] m/s²")
                    print(f"  Accel magnitude: {stats['accel']['magnitude_mean']:.3f} m/s²")
                
                # Reset counters
                frame_count = 0
                last_stats_time = current_time
                orientation_times = []
                print("=" * 50)
            
            # Small delay to prevent overwhelming output
            time.sleep(0.005)  # 5ms
            
    except KeyboardInterrupt:
        print(f"\n\nShutting down...")
        
        # Final orientation summary
        if estimator.orientation_history:
            print(f"\nFinal Orientation Analysis:")
            final_orientation = estimator.orientation
            final_euler = np.degrees(final_orientation.to_euler_angles())
            final_bias = estimator.gyro_bias
            
            print(f"Final orientation (RPY): [{final_euler[0]:+7.2f}, {final_euler[1]:+7.2f}, {final_euler[2]:+7.2f}] degrees")
            print(f"Final gyro bias: [{final_bias[0]:+8.5f}, {final_bias[1]:+8.5f}, {final_bias[2]:+8.5f}] rad/s")
            print(f"Total samples processed: {len(estimator.orientation_history)}")
            
            # Bias drift analysis
            if len(estimator.bias_history) > 100:
                bias_array = np.array(estimator.bias_history[-100:])  # Last 100 samples
                bias_std = np.std(bias_array, axis=0)
                print(f"Bias stability (std): [{bias_std[0]:.6f}, {bias_std[1]:.6f}, {bias_std[2]:.6f}] rad/s")
        
    finally:
        print("Stopping subscriber...")
        subscriber.stop()
        print("Done!")


if __name__ == "__main__":
    main()
