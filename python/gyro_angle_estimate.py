#!/usr/bin/env python3
"""
Simple gyroscope-only angle estimation program.
Uses pure gyro integration without complementary filtering to test basic functionality.
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
    Quaternion, load_calibration, IMUProcessor,
    create_axis_swap_matrix
)


class SimpleGyroIntegrator:
    """Simple gyroscope-only orientation integrator."""
    
    def __init__(self, gyro_frame_transform=None):
        """Initialize with optional coordinate transform."""
        self.orientation = Quaternion()  # Current orientation estimate
        self.gyro_transform = gyro_frame_transform if gyro_frame_transform is not None else np.eye(3)
        self.total_samples = 0
        self.integration_history = []  # Store recent orientations for analysis
        
    def apply_transform(self, gyro: np.ndarray) -> np.ndarray:
        """Apply coordinate frame transformation to gyro data."""
        return self.gyro_transform @ gyro
    
    def integrate_gyro(self, gyro: np.ndarray, dt: float) -> Quaternion:
        """Integrate gyroscope data to update orientation."""
        # Apply coordinate transform
        gyro_transformed = self.apply_transform(gyro)
        
        # Create quaternion from angular velocity
        delta_q = Quaternion.from_omega(gyro_transformed, dt)
        
        # Update orientation
        self.orientation = self.orientation * delta_q
        self.orientation.normalize()
        
        # Store history
        self.integration_history.append(self.orientation.q.copy())
        if len(self.integration_history) > 1000:  # Keep last 1000 samples
            self.integration_history = self.integration_history[-1000:]
        
        self.total_samples += 1
        return self.orientation
    
    def reset(self):
        """Reset orientation to identity."""
        self.orientation = Quaternion()
        self.integration_history = []
        print("Orientation reset to identity")
    
    def get_stats(self):
        """Get integration statistics."""
        if len(self.integration_history) < 2:
            return {"error": "Not enough data"}
        
        # Calculate recent orientation changes
        recent_orientations = np.array(self.integration_history[-50:])  # Last 50
        if len(recent_orientations) > 1:
            # Calculate orientation change rates
            diffs = np.diff(recent_orientations, axis=0)
            mean_change = np.mean(np.linalg.norm(diffs, axis=1))
        else:
            mean_change = 0.0
        
        current_euler = np.degrees(self.orientation.to_euler_angles())
        
        return {
            "total_samples": self.total_samples,
            "current_euler_deg": current_euler,
            "mean_orientation_change_rate": mean_change,
            "orientation_magnitude": np.linalg.norm(self.orientation.q)
        }


class GyroRerunVisualizer:
    """Rerun-based visualization for gyro-only integration."""
    
    def __init__(self, app_id: str = "gyro_angle_estimate"):
        """Initialize Rerun logging."""
        if not HAS_RERUN:
            print("Rerun not available - visualization disabled")
            return
            
        rr.init(app_id, spawn=True)
        
        # Set up the 3D view
        self.setup_world_frame()
        
        # Cube size
        self.cube_size = 0.3
        
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
    
    def log_orientation_cube(self, quaternion: Quaternion):
        """Log cube showing current orientation."""
        if not HAS_RERUN:
            return
        
        # Cube position
        pos = [0.0, 0.0, 0.5]
        color = [100, 150, 255, 200]  # Blue
        
        # Create cube vertices (unit cube centered at origin)
        s = self.cube_size / 2
        vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Bottom
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]       # Top
        ])
        
        # Rotate vertices by quaternion
        rotated_vertices = np.array([quaternion.rotate_vector(v) for v in vertices])
        
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
        rr.log("orientation/cube", rr.Mesh3D(
            vertex_positions=final_vertices,
            triangle_indices=triangles,
            albedo_factor=color
        ))
        
        # Log orientation as arrows for each axis
        # X-axis (red)
        x_forward = quaternion.rotate_vector(np.array([0.4, 0, 0]))
        rr.log("orientation/x_axis", rr.Arrows3D(
            origins=[pos],
            vectors=[x_forward],
            colors=[[255, 0, 0]]
        ))
        
        # Y-axis (green)
        y_forward = quaternion.rotate_vector(np.array([0, 0.4, 0]))
        rr.log("orientation/y_axis", rr.Arrows3D(
            origins=[pos],
            vectors=[y_forward],
            colors=[[0, 255, 0]]
        ))
        
        # Z-axis (blue)
        z_forward = quaternion.rotate_vector(np.array([0, 0, 0.4]))
        rr.log("orientation/z_axis", rr.Arrows3D(
            origins=[pos],
            vectors=[z_forward],
            colors=[[0, 0, 255]]
        ))
    
    def log_gyro_data(self, raw_gyro, transformed_gyro, timestamp):
        """Log gyroscope data for analysis."""
        if not HAS_RERUN:
            return
        
        rr.set_time_seconds("timestamp", timestamp)
        
        # Log raw gyro data
        rr.log("gyro/raw_magnitude", rr.Scalar(np.linalg.norm(raw_gyro)))
        rr.log("gyro/raw_x", rr.Scalar(raw_gyro[0]))
        rr.log("gyro/raw_y", rr.Scalar(raw_gyro[1]))
        rr.log("gyro/raw_z", rr.Scalar(raw_gyro[2]))
        
        # Log transformed gyro data
        rr.log("gyro/transformed_magnitude", rr.Scalar(np.linalg.norm(transformed_gyro)))
        rr.log("gyro/transformed_x", rr.Scalar(transformed_gyro[0]))
        rr.log("gyro/transformed_y", rr.Scalar(transformed_gyro[1]))
        rr.log("gyro/transformed_z", rr.Scalar(transformed_gyro[2]))
    
    def log_euler_angles(self, quaternion: Quaternion, timestamp):
        """Log Euler angles."""
        if not HAS_RERUN:
            return
        
        rr.set_time_seconds("timestamp", timestamp)
        
        # Convert to Euler angles (degrees)
        roll, pitch, yaw = quaternion.to_euler_angles()
        roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
        
        rr.log("euler/roll", rr.Scalar(roll_deg))
        rr.log("euler/pitch", rr.Scalar(pitch_deg))
        rr.log("euler/yaw", rr.Scalar(yaw_deg))


def main():
    """Main gyro integration loop."""
    print("=== Simple Gyroscope Angle Estimation ===")
    print("This program demonstrates pure gyro integration without complementary filtering.")
    print("Using coordinate transform: swap Y and Z axes")
    print()
    
    # Define the gyro transform (swap Y and Z based on testing)
    gyro_transform = create_axis_swap_matrix('x', 'z', 'y')  # Swap Y and Z
    
    print("Applied gyroscope coordinate transform (swap Y/Z):")
    print(gyro_transform)
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
    integrator = SimpleGyroIntegrator(gyro_frame_transform=gyro_transform)
    
    # Initialize visualization
    if HAS_RERUN:
        visualizer = GyroRerunVisualizer()
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
    integration_times = []
    
    print("\nControls:")
    print("- Ctrl+C: Quit")
    print("- The program will auto-reset if orientation seems to drift too much")
    print("Starting integration...\n")
    
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
            
            # Apply calibration (only need gyro)
            cal_gyro, _ = processor.process_sample(raw_gyro, raw_accel)
            
            # Calculate dt
            if last_timestamp is not None:
                dt = (timestamp - last_timestamp) / 1000.0  # Convert ms to seconds
            else:
                dt = 0.01  # Initial estimate: 10ms
            last_timestamp = timestamp
            
            # Gyro integration
            integration_start = time.perf_counter()
            
            # Get transformed gyro for display
            transformed_gyro = integrator.apply_transform(cal_gyro)
            
            # Integrate gyroscope
            current_orientation = integrator.integrate_gyro(cal_gyro, dt)
            
            integration_time = time.perf_counter() - integration_start
            integration_times.append(integration_time)
            
            # Visualization
            if visualizer is not None:
                visualizer.log_orientation_cube(current_orientation)
                visualizer.log_gyro_data(cal_gyro, transformed_gyro, timestamp)
                visualizer.log_euler_angles(current_orientation, timestamp)
            
            # Print periodic status
            if frame_count % 50 == 0:  # Every 50 frames
                # Convert to Euler angles for display
                euler_angles = np.degrees(current_orientation.to_euler_angles())
                
                print(f"\n--- Frame {frame_count} ---")
                print(f"Timestamp: {timestamp:.3f}, dt: {dt:.4f}s")
                print(f"Raw Gyro:         [{raw_gyro[0]:+7.4f}, {raw_gyro[1]:+7.4f}, {raw_gyro[2]:+7.4f}] rad/s")
                print(f"Calibrated Gyro:  [{cal_gyro[0]:+7.4f}, {cal_gyro[1]:+7.4f}, {cal_gyro[2]:+7.4f}] rad/s")
                print(f"Transformed Gyro: [{transformed_gyro[0]:+7.4f}, {transformed_gyro[1]:+7.4f}, {transformed_gyro[2]:+7.4f}] rad/s")
                print(f"Gyro magnitude:   {np.linalg.norm(cal_gyro):.4f} rad/s")
                
                print(f"\nOrientation (Roll, Pitch, Yaw in degrees):")
                print(f"  [{euler_angles[0]:+7.2f}, {euler_angles[1]:+7.2f}, {euler_angles[2]:+7.2f}]")
                
                # Check for potential issues
                gyro_mag = np.linalg.norm(cal_gyro)
                if gyro_mag > 0.5:
                    print(f"  ⚠ High gyro activity: {gyro_mag:.3f} rad/s")
                elif gyro_mag < 0.001:
                    print(f"  ℹ Very low gyro activity: {gyro_mag:.6f} rad/s")
                else:
                    print(f"  ✓ Normal gyro activity")
                
                # Check orientation magnitude (should be ~1 for unit quaternion)
                quat_mag = np.linalg.norm(current_orientation.q)
                if abs(quat_mag - 1.0) > 0.01:
                    print(f"  ⚠ Quaternion magnitude: {quat_mag:.6f} (should be ~1.0)")
                else:
                    print(f"  ✓ Quaternion normalized properly")
            
            # Auto-reset if orientation seems problematic
            quat_magnitude = np.linalg.norm(current_orientation.q)
            if abs(quat_magnitude - 1.0) > 0.1 or frame_count % 1000 == 0:  # Reset every 1000 frames or if quaternion is bad
                if abs(quat_magnitude - 1.0) > 0.1:
                    print(f"⚠ Quaternion magnitude error: {quat_magnitude:.6f}, resetting...")
                elif frame_count % 1000 == 0:
                    print(f"Periodic reset at frame {frame_count}")
                integrator.reset()
            
            # Print statistics every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:
                elapsed = current_time - last_stats_time
                fps = frame_count / elapsed
                
                if integration_times:
                    avg_integration_time = sum(integration_times) / len(integration_times) * 1000
                    max_integration_time = max(integration_times) * 1000
                
                print(f"\n=== STATISTICS (last {elapsed:.1f}s) ===")
                print(f"Frame rate: {fps:.2f} FPS")
                print(f"Integration processing: {avg_integration_time:.3f}ms avg, {max_integration_time:.3f}ms max")
                print(subscriber.get_stats())
                
                # Integration statistics
                stats = integrator.get_stats()
                if "error" not in stats:
                    print(f"\nGyro Integration Stats:")
                    print(f"  Total samples: {stats['total_samples']}")
                    print(f"  Current orientation (RPY): [{stats['current_euler_deg'][0]:+7.2f}, {stats['current_euler_deg'][1]:+7.2f}, {stats['current_euler_deg'][2]:+7.2f}] degrees")
                    print(f"  Quaternion magnitude: {stats['orientation_magnitude']:.6f}")
                    print(f"  Mean orientation change rate: {stats['mean_orientation_change_rate']:.6f}")
                
                # IMU statistics
                imu_stats = processor.get_statistics(use_calibrated=True)
                if "error" not in imu_stats:
                    print(f"\nCalibrated Gyro Stats ({imu_stats['num_samples']} samples):")
                    print(f"  Mean:  [{imu_stats['gyro']['mean'][0]:+8.5f}, {imu_stats['gyro']['mean'][1]:+8.5f}, {imu_stats['gyro']['mean'][2]:+8.5f}] rad/s")
                    print(f"  Std:   [{imu_stats['gyro']['std'][0]:+8.5f}, {imu_stats['gyro']['std'][1]:+8.5f}, {imu_stats['gyro']['std'][2]:+8.5f}] rad/s")
                    print(f"  Magnitude: {imu_stats['gyro']['magnitude_mean']:.5f} rad/s")
                
                # Reset counters
                frame_count = 0
                last_stats_time = current_time
                integration_times = []
                print("=" * 50)
            
            # Small delay to prevent overwhelming output
            time.sleep(0.005)  # 5ms
            
    except KeyboardInterrupt:
        print(f"\n\nShutting down...")
        
        # Final summary
        stats = integrator.get_stats()
        if "error" not in stats:
            print(f"\nFinal Integration Summary:")
            print(f"Total samples processed: {stats['total_samples']}")
            print(f"Final orientation (RPY): [{stats['current_euler_deg'][0]:+7.2f}, {stats['current_euler_deg'][1]:+7.2f}, {stats['current_euler_deg'][2]:+7.2f}] degrees")
            print(f"Final quaternion magnitude: {stats['orientation_magnitude']:.6f}")
        
    finally:
        print("Stopping subscriber...")
        subscriber.stop()
        print("Done!")


if __name__ == "__main__":
    main() 