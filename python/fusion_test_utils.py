#!/usr/bin/env python3
"""
Test utilities for sensor fusion system.
Provides synthetic data generation and testing capabilities.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass

from fusion_pose_estimate import (
    IMUMeasurement, VisualOdometryMeasurement, 
    SensorInterface, SensorMeasurement
)
from imu_utils import Quaternion


class SyntheticTrajectory:
    """Generate synthetic trajectory data for testing."""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.t = 0.0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = Quaternion()
        self.angular_velocity = np.zeros(3)
        
    def circular_motion(self, radius: float = 1.0, omega: float = 0.5) -> Tuple[np.ndarray, Quaternion]:
        """Generate circular motion trajectory."""
        # Position on circle
        x = radius * np.cos(omega * self.t)
        y = radius * np.sin(omega * self.t)
        z = 0.0
        
        # Velocity
        vx = -radius * omega * np.sin(omega * self.t)
        vy = radius * omega * np.cos(omega * self.t)
        vz = 0.0
        
        # Orientation (facing direction of motion)
        yaw = omega * self.t + np.pi/2
        orientation = Quaternion.from_euler(0, 0, yaw)
        
        # Angular velocity (turning rate)
        angular_vel = np.array([0, 0, omega])
        
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.orientation = orientation
        self.angular_velocity = angular_vel
        
        self.t += self.dt
        
        return self.position.copy(), orientation
        
    def sinusoidal_motion(self, amplitude: float = 1.0, frequency: float = 0.2) -> Tuple[np.ndarray, Quaternion]:
        """Generate sinusoidal motion along X axis with rotation."""
        # Position
        x = self.t * 0.1  # Forward motion
        y = amplitude * np.sin(2 * np.pi * frequency * self.t)
        z = 0.0
        
        # Velocity
        vx = 0.1
        vy = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * self.t)
        vz = 0.0
        
        # Orientation (slight pitch and roll)
        roll = 0.1 * np.sin(2 * np.pi * frequency * self.t)
        pitch = 0.05 * np.sin(2 * np.pi * frequency * self.t * 2)
        yaw = 0.0
        orientation = Quaternion.from_euler(roll, pitch, yaw)
        
        # Angular velocity
        angular_vel = np.array([
            0.1 * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * self.t),
            0.05 * 2 * np.pi * frequency * 2 * np.cos(2 * np.pi * frequency * self.t * 2),
            0.0
        ])
        
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.orientation = orientation
        self.angular_velocity = angular_vel
        
        self.t += self.dt
        
        return self.position.copy(), orientation
        
    def get_imu_measurement(self, noise_gyro: float = 0.01, noise_accel: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate IMU measurements from current state."""
        # Gyroscope measurement (angular velocity in body frame)
        gyro = self.angular_velocity + np.random.randn(3) * noise_gyro
        
        # Accelerometer measurement (includes gravity)
        # Transform gravity to body frame
        gravity_world = np.array([0, 0, -9.81])
        R = self.orientation.to_rotation_matrix()
        gravity_body = R.T @ gravity_world
        
        # Add linear acceleration (simplified - ignores centripetal acceleration)
        accel = -gravity_body + np.random.randn(3) * noise_accel
        
        return gyro, accel


class SyntheticIMUSensor(SensorInterface):
    """Synthetic IMU sensor for testing."""
    
    def __init__(self, trajectory: SyntheticTrajectory, imu_rate: float = 100.0):
        self.trajectory = trajectory
        self.imu_rate = imu_rate
        self.imu_dt = 1.0 / imu_rate
        self.last_time = time.time()
        self.frame_id = 0
        self.start_time = time.time() * 1000  # ms
        
    def get_measurement(self) -> Optional[IMUMeasurement]:
        """Get synthetic IMU measurement."""
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_time < self.imu_dt:
            return None
            
        # Generate trajectory
        if self.frame_id % 1000 < 500:
            # First half: circular motion
            pos, orient = self.trajectory.circular_motion(radius=2.0, omega=0.3)
        else:
            # Second half: sinusoidal motion
            pos, orient = self.trajectory.sinusoidal_motion(amplitude=0.5, frequency=0.2)
            
        # Generate IMU measurement
        gyro, accel = self.trajectory.get_imu_measurement()
        
        self.frame_id += 1
        self.last_time = current_time
        timestamp = time.time() * 1000  # ms
        
        return IMUMeasurement(
            timestamp=timestamp,
            frame_id=self.frame_id,
            gyro=gyro,
            accel=accel,
            dt=self.imu_dt
        )
        
    def is_ready(self) -> bool:
        """Always ready for synthetic data."""
        return True


class SyntheticVisualOdometrySensor(SensorInterface):
    """Synthetic visual odometry sensor for testing."""
    
    def __init__(self, trajectory: SyntheticTrajectory, vo_rate: float = 10.0,
                 noise_translation: float = 0.02, noise_rotation: float = 0.05):
        self.trajectory = trajectory
        self.vo_rate = vo_rate
        self.vo_dt = 1.0 / vo_rate
        self.last_time = time.time()
        self.frame_id = 0
        
        # Noise parameters
        self.noise_translation = noise_translation
        self.noise_rotation = noise_rotation
        
        # Previous state for computing relative transform
        self.prev_position = None
        self.prev_orientation = None
        
    def get_measurement(self) -> Optional[VisualOdometryMeasurement]:
        """Get synthetic visual odometry measurement."""
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_time < self.vo_dt:
            return None
            
        # Generate trajectory (must match IMU trajectory)
        if self.frame_id % 100 < 50:
            # Circular motion
            pos, orient = self.trajectory.circular_motion(radius=2.0, omega=0.3)
        else:
            # Sinusoidal motion  
            pos, orient = self.trajectory.sinusoidal_motion(amplitude=0.5, frequency=0.2)
            
        # Compute relative transform
        if self.prev_position is None:
            self.prev_position = pos
            self.prev_orientation = orient
            return None
            
        # Relative translation
        delta_pos = pos - self.prev_position
        
        # Relative rotation
        delta_orientation = self.prev_orientation.inverse() * orient
        delta_R = delta_orientation.to_rotation_matrix()
        
        # Add noise
        delta_pos += np.random.randn(3) * self.noise_translation
        
        # Add noise to rotation (small angle approximation)
        noise_angles = np.random.randn(3) * self.noise_rotation
        noise_R = Quaternion.from_euler(*noise_angles).to_rotation_matrix()
        delta_R = delta_R @ noise_R
        
        # Construct SE(3) transform
        T = np.eye(4)
        T[:3, :3] = delta_R
        T[:3, 3] = delta_pos
        
        # Generate synthetic 3D points (random points in view)
        num_points = np.random.randint(50, 200)
        points_prev = np.random.randn(num_points, 3) * 2.0
        points_prev[:, 2] = np.abs(points_prev[:, 2]) + 1.0  # Ensure positive depth
        
        # Transform points
        points_curr = (delta_R @ points_prev.T).T + delta_pos
        
        # Covariance based on number of points
        base_cov = np.eye(6) * 0.01
        cov_scale = max(1.0, 100.0 / num_points)
        covariance = base_cov * cov_scale
        
        # Update previous state
        self.prev_position = pos
        self.prev_orientation = orient
        
        self.frame_id += 1
        self.last_time = current_time
        timestamp = time.time() * 1000  # ms
        
        return VisualOdometryMeasurement(
            timestamp=timestamp,
            frame_id=self.frame_id,
            transform=T,
            covariance=covariance,
            num_inliers=num_points,
            points_prev=points_prev,
            points_curr=points_curr
        )
        
    def is_ready(self) -> bool:
        """Always ready for synthetic data."""
        return True


def test_sensor_fusion_with_synthetic_data():
    """Test the sensor fusion system with synthetic data."""
    from fusion_pose_estimate import SensorFusion, FusionVisualizer
    import threading
    
    print("=== Testing Sensor Fusion with Synthetic Data ===")
    
    # Create synthetic trajectory generator
    trajectory = SyntheticTrajectory(dt=0.01)
    
    # Create synthetic sensors
    imu_sensor = SyntheticIMUSensor(trajectory, imu_rate=100.0)
    vo_sensor = SyntheticVisualOdometrySensor(trajectory, vo_rate=10.0)
    
    # Create visualizer
    visualizer = FusionVisualizer("fusion_test_synthetic", spawn=True)
    
    # Create fusion system
    fusion = SensorFusion(imu_sensor, vo_sensor, visualizer, fusion_rate_hz=50)
    
    print("Starting synthetic data fusion test...")
    print("The system will generate:")
    print("- Circular motion for first 5 seconds")
    print("- Sinusoidal motion for next 5 seconds")
    print("- And repeat...")
    print("\nPress Ctrl+C to stop")
    
    # Start fusion in separate thread
    fusion_thread = threading.Thread(target=fusion.run, daemon=True)
    fusion_thread.start()
    
    # Monitor and print statistics
    last_stats_time = time.time()
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_stats_time >= 2.0:  # Print every 2 seconds
                state = fusion.get_current_state()
                
                if "error" not in state:
                    print(f"\n=== Synthetic Test @ t={elapsed:.1f}s ===")
                    
                    # Current motion type
                    motion_type = "Circular" if int(elapsed * 10) % 100 < 50 else "Sinusoidal"
                    print(f"Motion type: {motion_type}")
                    
                    # State
                    print(f"Position: [{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}] m")
                    print(f"Orientation (RPY): [{state['euler_degrees'][0]:.1f}, {state['euler_degrees'][1]:.1f}, {state['euler_degrees'][2]:.1f}] deg")
                    
                    # Stats
                    stats = state['stats']
                    print(f"\nRates:")
                    if stats['imu_count'] > 0:
                        imu_rate = stats['imu_count'] / (current_time - last_stats_time)
                        print(f"  IMU: {imu_rate:.1f} Hz (target: 100 Hz)")
                    if stats['vo_count'] > 0:
                        vo_rate = stats['vo_count'] / (current_time - last_stats_time)
                        print(f"  VO: {vo_rate:.1f} Hz (target: 10 Hz)")
                    if stats['fusion_count'] > 0:
                        fusion_rate = stats['fusion_count'] / (current_time - last_stats_time)
                        print(f"  Fusion: {fusion_rate:.1f} Hz (target: 50 Hz)")
                        
                    # Reset counters
                    fusion.stats['imu_count'] = 0
                    fusion.stats['vo_count'] = 0  
                    fusion.stats['fusion_count'] = 0
                    
                print("=" * 50)
                last_stats_time = current_time
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest completed!")
        
        
if __name__ == "__main__":
    test_sensor_fusion_with_synthetic_data() 