#!/usr/bin/env python3
"""
IMU utility functions for calibration, offset correction, and data processing.
"""

import numpy as np
import json
import os
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
import time


@dataclass
class IMUCalibration:
    """Container for IMU calibration parameters."""
    gyro_offset: np.ndarray  # 3D gyro bias (rad/s)
    accel_offset: np.ndarray  # 3D accel bias (m/s²)
    gravity_vector: np.ndarray  # Expected gravity vector (m/s²)
    timestamp: float  # When calibration was performed
    num_samples: int  # Number of samples used for calibration
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'gyro_offset': self.gyro_offset.tolist(),
            'accel_offset': self.accel_offset.tolist(),
            'gravity_vector': self.gravity_vector.tolist(),
            'timestamp': self.timestamp,
            'num_samples': self.num_samples
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IMUCalibration':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            gyro_offset=np.array(data['gyro_offset']),
            accel_offset=np.array(data['accel_offset']),
            gravity_vector=np.array(data['gravity_vector']),
            timestamp=data['timestamp'],
            num_samples=data['num_samples']
        )


class Quaternion:
    """Unit quaternion class for SO(3) operations."""
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize quaternion with w + xi + yj + zk."""
        self.q = np.array([w, x, y, z], dtype=np.float64)
        self.normalize()
    
    @property
    def w(self) -> float:
        """Scalar (real) part of quaternion."""
        return self.q[0]
    
    @property
    def x(self) -> float:
        """X component of vector part."""
        return self.q[1]
    
    @property
    def y(self) -> float:
        """Y component of vector part."""
        return self.q[2]
    
    @property
    def z(self) -> float:
        """Z component of vector part."""
        return self.q[3]
    
    @w.setter
    def w(self, value: float):
        """Set scalar part."""
        self.q[0] = value
    
    @x.setter
    def x(self, value: float):
        """Set X component."""
        self.q[1] = value
    
    @y.setter
    def y(self, value: float):
        """Set Y component."""
        self.q[2] = value
    
    @z.setter
    def z(self, value: float):
        """Set Z component."""
        self.q[3] = value
    
    @classmethod
    def from_array(cls, q: np.ndarray) -> 'Quaternion':
        """Create from numpy array [w, x, y, z]."""
        quat = cls()
        quat.q = q.copy()
        quat.normalize()
        return quat
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """Create from axis-angle representation."""
        if np.linalg.norm(axis) == 0:
            return cls()  # Identity
        
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis
        return cls(w, xyz[0], xyz[1], xyz[2])
    
    @classmethod
    def from_scaled_axis(cls, omega: np.ndarray) -> 'Quaternion':
        """Create from scaled axis (exponential map: omega = axis * angle)."""
        angle = np.linalg.norm(omega)
        if angle < 1e-8:
            return cls()  # Identity for small angles
        axis = omega / angle
        return cls.from_axis_angle(axis, angle)
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """Create quaternion from Euler angles (roll, pitch, yaw) in radians.
        
        Uses the ZYX convention (yaw-pitch-roll order):
        q = q_yaw * q_pitch * q_roll
        """
        # Half angles
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        # Quaternion multiplication: q_yaw * q_pitch * q_roll
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(w, x, y, z)
    
    def normalize(self) -> 'Quaternion':
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(self.q)
        if norm > 1e-8:
            self.q = self.q / norm
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Default to identity
        return self
    
    def conjugate(self) -> 'Quaternion':
        """Return quaternion conjugate."""
        return Quaternion.from_array(np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]))
    
    def inverse(self) -> 'Quaternion':
        """Return quaternion inverse (same as conjugate for unit quaternions)."""
        return self.conjugate()
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication."""
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return Quaternion(w, x, y, z)
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate vector by this quaternion."""
        # v' = q * [0, v] * q*
        v_quat = Quaternion(0, v[0], v[1], v[2])
        result = self * v_quat * self.conjugate()
        return result.q[1:4]
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self.q
        
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    def to_euler_angles(self) -> Tuple[float, float, float]:
        """Convert to Euler angles (roll, pitch, yaw) in radians."""
        w, x, y, z = self.q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    @classmethod
    def from_omega(cls, omega: np.ndarray, dt: float) -> "Quaternion":
        """Exponential map: q = exp(½·ω·dt)."""
        theta = np.linalg.norm(omega) * dt
        if theta < 1e-8:                               # small‑angle shortcut
            return cls()                               # identity
        axis  = omega / np.linalg.norm(omega)
        half  = 0.5 * theta
        return cls(np.cos(half), *(np.sin(half)*axis))

class IMUProcessor:
    """IMU data processor with calibration support."""
    
    def __init__(self, calibration: Optional[IMUCalibration] = None):
        """Initialize with optional calibration data."""
        self.calibration = calibration
        self.raw_gyro_history = []
        self.raw_accel_history = []
        self.calibrated_gyro_history = []
        self.calibrated_accel_history = []
    
    def apply_calibration(self, gyro: np.ndarray, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply calibration to raw IMU data."""
        if self.calibration is None:
            return gyro, accel
        
        # Apply gyro offset correction
        calibrated_gyro = gyro - self.calibration.gyro_offset
        
        # Apply accel offset correction BUT KEEP GRAVITY for orientation filter
        # Only remove the bias offset, not the gravity vector
        calibrated_accel = accel - self.calibration.accel_offset
        
        return calibrated_gyro, calibrated_accel
    
    def apply_calibration_remove_gravity(self, gyro: np.ndarray, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply calibration and remove gravity (for motion analysis)."""
        if self.calibration is None:
            return gyro, accel
        
        # Apply gyro offset correction
        calibrated_gyro = gyro - self.calibration.gyro_offset
        
        # Apply accel offset correction and gravity compensation
        calibrated_accel = accel - self.calibration.accel_offset - self.calibration.gravity_vector
        
        return calibrated_gyro, calibrated_accel
    
    def process_sample(self, gyro: np.ndarray, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single IMU sample and store history."""
        # Store raw data
        self.raw_gyro_history.append(gyro.copy())
        self.raw_accel_history.append(accel.copy())
        
        # Apply calibration (keeping gravity for orientation filter)
        cal_gyro, cal_accel = self.apply_calibration(gyro, accel)
        
        # Store calibrated data
        self.calibrated_gyro_history.append(cal_gyro.copy())
        self.calibrated_accel_history.append(cal_accel.copy())
        
        # Keep only last 1000 samples to prevent memory growth
        if len(self.raw_gyro_history) > 1000:
            self.raw_gyro_history = self.raw_gyro_history[-1000:]
            self.raw_accel_history = self.raw_accel_history[-1000:]
            self.calibrated_gyro_history = self.calibrated_gyro_history[-1000:]
            self.calibrated_accel_history = self.calibrated_accel_history[-1000:]
        
        return cal_gyro, cal_accel
    
    def get_statistics(self, use_calibrated: bool = True) -> Dict:
        """Get statistics for recent IMU data."""
        if use_calibrated and self.calibrated_gyro_history:
            gyro_data = np.array(self.calibrated_gyro_history[-100:])  # Last 100 samples
            accel_data = np.array(self.calibrated_accel_history[-100:])
            data_type = "calibrated"
        elif self.raw_gyro_history:
            gyro_data = np.array(self.raw_gyro_history[-100:])
            accel_data = np.array(self.raw_accel_history[-100:])
            data_type = "raw"
        else:
            return {"error": "No data available"}
        
        return {
            "data_type": data_type,
            "num_samples": len(gyro_data),
            "gyro": {
                "mean": np.mean(gyro_data, axis=0),
                "std": np.std(gyro_data, axis=0),
                "magnitude_mean": np.mean(np.linalg.norm(gyro_data, axis=1))
            },
            "accel": {
                "mean": np.mean(accel_data, axis=0),
                "std": np.std(accel_data, axis=0),
                "magnitude_mean": np.mean(np.linalg.norm(accel_data, axis=1))
            }
        }


def collect_calibration_samples(subscriber, num_samples: int = 50, 
                              timeout_seconds: float = 30.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Collect IMU samples for calibration while device is stationary.
    
    Args:
        subscriber: ZenohD435iSubscriber instance
        num_samples: Number of samples to collect
        timeout_seconds: Max time to wait for samples
    
    Returns:
        Tuple of (gyro_samples, accel_samples) as lists of numpy arrays
    """
    print(f"Collecting {num_samples} calibration samples...")
    print("Please keep the device STATIONARY during calibration!")
    
    gyro_samples = []
    accel_samples = []
    last_frame_id = -1
    start_time = time.time()
    
    while len(gyro_samples) < num_samples:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Failed to collect {num_samples} samples within {timeout_seconds} seconds")
        
        # Get latest frame
        frame_data = subscriber.get_latest_frames()
        
        if frame_data.frame_count == 0 or frame_data.frame_count == last_frame_id:
            time.sleep(0.001)
            continue
        
        if frame_data.motion is None:
            time.sleep(0.001)
            continue
        
        last_frame_id = frame_data.frame_count
        
        # Extract IMU data
        gyro = np.array(frame_data.motion.gyro)
        accel = np.array(frame_data.motion.accel)
        
        gyro_samples.append(gyro)
        accel_samples.append(accel)
        
        # Progress indicator
        if len(gyro_samples) % 10 == 0:
            print(f"  Collected {len(gyro_samples)}/{num_samples} samples...")
    
    print(f"✓ Collected {len(gyro_samples)} samples in {time.time() - start_time:.2f} seconds")
    return gyro_samples, accel_samples


def compute_calibration(gyro_samples: List[np.ndarray], 
                       accel_samples: List[np.ndarray],
                       gravity_magnitude: float = 9.81) -> IMUCalibration:
    """
    Compute IMU calibration from collected samples.
    
    Args:
        gyro_samples: List of gyro readings (rad/s)
        accel_samples: List of accel readings (m/s²)
        gravity_magnitude: Expected gravity magnitude (m/s²)
    
    Returns:
        IMUCalibration object
    """
    gyro_array = np.array(gyro_samples)
    accel_array = np.array(accel_samples)
    
    # Gyro offset is just the mean (should be ~0 when stationary)
    gyro_offset = np.mean(gyro_array, axis=0)
    
    # Accel offset: subtract expected gravity from mean
    accel_mean = np.mean(accel_array, axis=0)
    
    # Determine gravity direction (largest magnitude component)
    gravity_axis = np.argmax(np.abs(accel_mean))
    gravity_vector = np.zeros(3)
    gravity_vector[gravity_axis] = np.sign(accel_mean[gravity_axis]) * gravity_magnitude
    
    # Accel offset is the deviation from expected gravity
    accel_offset = accel_mean - gravity_vector
    
    calibration = IMUCalibration(
        gyro_offset=gyro_offset,
        accel_offset=accel_offset,
        gravity_vector=gravity_vector,
        timestamp=time.time(),
        num_samples=len(gyro_samples)
    )
    
    print(f"\nCalibration computed:")
    print(f"  Gyro offset (rad/s):  [{gyro_offset[0]:+8.5f}, {gyro_offset[1]:+8.5f}, {gyro_offset[2]:+8.5f}]")
    print(f"  Accel offset (m/s²):  [{accel_offset[0]:+8.5f}, {accel_offset[1]:+8.5f}, {accel_offset[2]:+8.5f}]")
    print(f"  Gravity vector (m/s²): [{gravity_vector[0]:+8.5f}, {gravity_vector[1]:+8.5f}, {gravity_vector[2]:+8.5f}]")
    print(f"  Gravity axis: {['X', 'Y', 'Z'][gravity_axis]} ({gravity_vector[gravity_axis]:+.2f} m/s²)")
    
    return calibration


def save_calibration(calibration: IMUCalibration, filename: str = "imu_calibration.json") -> None:
    """Save calibration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(calibration.to_dict(), f, indent=2)
    print(f"✓ Calibration saved to {filename}")


def load_calibration(filename: str = "imu_calibration.json") -> Optional[IMUCalibration]:
    """Load calibration from JSON file."""
    if not os.path.exists(filename):
        print(f"Calibration file {filename} not found")
        return None
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        calibration = IMUCalibration.from_dict(data)
        print(f"✓ Calibration loaded from {filename}")
        return calibration
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None


def validate_calibration_quality(gyro_samples: List[np.ndarray], 
                                accel_samples: List[np.ndarray]) -> Dict:
    """
    Validate the quality of calibration samples.
    
    Returns:
        Dictionary with quality metrics and recommendations
    """
    gyro_array = np.array(gyro_samples)
    accel_array = np.array(accel_samples)
    
    # Calculate stability metrics
    gyro_std = np.std(gyro_array, axis=0)
    accel_std = np.std(accel_array, axis=0)
    
    gyro_max_std = np.max(gyro_std)
    accel_max_std = np.max(accel_std)
    
    # Quality thresholds
    gyro_good_threshold = 0.01  # rad/s
    accel_good_threshold = 0.1   # m/s²
    
    quality = {
        "gyro_stability": {
            "std_dev": gyro_std,
            "max_std": gyro_max_std,
            "is_good": gyro_max_std < gyro_good_threshold,
            "threshold": gyro_good_threshold
        },
        "accel_stability": {
            "std_dev": accel_std,
            "max_std": accel_max_std,
            "is_good": accel_max_std < accel_good_threshold,
            "threshold": accel_good_threshold
        },
        "overall_quality": "good" if (gyro_max_std < gyro_good_threshold and accel_max_std < accel_good_threshold) else "poor"
    }
    
    return quality


def print_quality_report(quality: Dict) -> None:
    """Print a human-readable quality report."""
    print(f"\nCalibration Quality Report:")
    print(f"  Overall Quality: {quality['overall_quality'].upper()}")
    
    gyro = quality['gyro_stability']
    print(f"  Gyro Stability: {'✓ GOOD' if gyro['is_good'] else '✗ POOR'}")
    print(f"    Max std dev: {gyro['max_std']:.6f} rad/s (threshold: {gyro['threshold']:.6f})")
    
    accel = quality['accel_stability']
    print(f"  Accel Stability: {'✓ GOOD' if accel['is_good'] else '✗ POOR'}")
    print(f"    Max std dev: {accel['max_std']:.6f} m/s² (threshold: {accel['threshold']:.6f})")
    
    if quality['overall_quality'] == 'poor':
        print(f"\n  Recommendations:")
        if not gyro['is_good']:
            print(f"    - Device was moving during gyro calibration. Keep it perfectly still.")
        if not accel['is_good']:
            print(f"    - Device was vibrating during accel calibration. Use a stable surface.")
        print(f"    - Consider recalibrating with device completely stationary.")


# Utility functions for creating common coordinate transforms
def create_axis_swap_matrix(x_axis: str, y_axis: str, z_axis: str) -> np.ndarray:
    """
    Create a matrix to swap/negate coordinate axes.
    
    Args:
        x_axis: What the new X axis should be (e.g., 'x', '-x', 'y', '-y', 'z', '-z')
        y_axis: What the new Y axis should be
        z_axis: What the new Z axis should be
    
    Returns:
        3x3 transformation matrix
    """
    axis_map = {
        'x': np.array([1, 0, 0]),
        '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]),
        '-y': np.array([0, -1, 0]),
        'z': np.array([0, 0, 1]),
        '-z': np.array([0, 0, -1])
    }
    
    return np.column_stack([axis_map[x_axis], axis_map[y_axis], axis_map[z_axis]])

def create_common_transforms():
    """Create some common coordinate frame transforms."""
    transforms = {
        'identity': np.eye(3),
        'negate_x': create_axis_swap_matrix('-x', 'y', 'z'),
        'negate_y': create_axis_swap_matrix('x', '-y', 'z'),
        'negate_z': create_axis_swap_matrix('x', 'y', '-z'),
        'swap_yz': create_axis_swap_matrix('x', 'z', 'y'),
        'swap_xz': create_axis_swap_matrix('z', 'y', 'x'),
        'swap_xy': create_axis_swap_matrix('y', 'x', 'z'),
        # Common IMU frame corrections
        'ned_to_enu': create_axis_swap_matrix('y', 'x', '-z'),  # North-East-Down to East-North-Up
        'rh_to_lh': create_axis_swap_matrix('x', 'z', 'y'),     # Right-hand to left-hand coordinate system
    }
    return transforms 

@dataclass
class MahonyState:
    q:  Quaternion          = Quaternion()     # Attitude estimate A←E
    b:  np.ndarray          = field(default_factory=lambda: np.zeros(3))      # Gyro bias estimate (rad/s)
    e_int: np.ndarray       = field(default_factory=lambda: np.zeros(3))      # Optional integral of error

# --------------------------- Mahony filter -----------------------------------
class MahonyFilter:
    """
    Passive non‑linear complementary filter on SO(3) (Mahony 2005, Eq. 11–13).
    Only gyro (rad/s) and accel (m/s²) are required; magnetometer is optional.
    """
    def __init__(self, k_p: float = 2.0, k_i: float = 0.1,
                 gravity_mag: float = 9.81, estimate_bias: bool = True,
                 accel_frame_transform: Optional[np.ndarray] = None,
                 gyro_frame_transform: Optional[np.ndarray] = None):
        self.k_p   = k_p
        self.k_i   = k_i if estimate_bias else 0.0
        self.g_mag = gravity_mag
        self.state = MahonyState()
        
        # Coordinate frame transforms (default: identity - no change)
        self.accel_transform = accel_frame_transform if accel_frame_transform is not None else np.eye(3)
        self.gyro_transform = gyro_frame_transform if gyro_frame_transform is not None else np.eye(3)
        
        # For compatibility with legacy interface - track separate orientations
        self.gyro_only_orientation = Quaternion()
        self.accel_only_orientation = Quaternion()
        
        # Track initialization state
        self.is_initialized = False
        
        # Debug tracking for compatibility with legacy interface
        self.total_updates = 0
        self.accel_used_count = 0
        self.accel_rejected_count = 0
        self.gravity_estimates = []

    def apply_coordinate_transforms(self, gyro: np.ndarray, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply coordinate frame transformations to sensor data."""
        transformed_gyro = self.gyro_transform @ gyro
        transformed_accel = self.accel_transform @ accel
        return transformed_gyro, transformed_accel

    @property
    def gyro_bias(self) -> np.ndarray:
        """Get current gyro bias estimate for compatibility with legacy interface."""
        return self.state.b.copy()

    @property
    def gravity_mag(self) -> float:
        """Get gravity magnitude for compatibility with legacy interface."""
        return self.g_mag

    def _update_gyro_only_orientation(self, gyro_transformed: np.ndarray, dt: float):
        """Update gyro-only orientation estimate."""
        # Remove bias estimate
        omega = gyro_transformed - self.state.b
        
        # Use same integration method as main filter for consistency
        delta_q = Quaternion.from_omega(omega, dt)
        
        # Update gyro-only orientation
        self.gyro_only_orientation = self.gyro_only_orientation * delta_q
        self.gyro_only_orientation.normalize()

    def _update_accel_only_orientation(self, accel_transformed: np.ndarray):
        """Update accelerometer-only orientation estimate."""
        accel_norm = np.linalg.norm(accel_transformed)
        if accel_norm < 1e-6:
            return  # Keep previous if no accel data
        
        # Normalize accelerometer reading
        accel_unit = accel_transformed / accel_norm
        
        # Expected gravity in world frame (pointing down)
        gravity_world = np.array([0.0, 0.0, -1.0])
        
        # Find rotation that aligns gravity_world with measured accel
        # Cross product gives rotation axis
        cross = np.cross(gravity_world, -accel_unit)  # Negative because accel points up when stationary
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-6:
            # Vectors are aligned, no rotation needed (or 180 degree flip)
            dot = np.dot(gravity_world, -accel_unit)
            if dot > 0:
                self.accel_only_orientation = Quaternion()  # Identity
            else:
                # 180 degree rotation around any perpendicular axis
                self.accel_only_orientation = Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
        else:
            # General case
            axis = cross / cross_norm
            angle = np.arccos(np.clip(np.dot(gravity_world, -accel_unit), -1.0, 1.0))
            self.accel_only_orientation = Quaternion.from_axis_angle(axis, angle)

    def get_all_orientations(self) -> Tuple[Quaternion, Quaternion, Quaternion]:
        """Get all three orientation estimates for comparison."""
        return (
            self.gyro_only_orientation,
            self.accel_only_orientation, 
            self.state.q
        )

    def get_debug_stats(self) -> Dict:
        """Get debug statistics about accelerometer usage for compatibility with legacy interface."""
        if self.total_updates > 0:
            accel_usage_rate = self.accel_used_count / self.total_updates * 100
            accel_rejection_rate = self.accel_rejected_count / self.total_updates * 100
        else:
            accel_usage_rate = 0
            accel_rejection_rate = 0
        
        # Compute average gravity direction estimate
        avg_gravity = np.array([0, 0, -1])  # Default
        if self.gravity_estimates:
            avg_gravity = np.mean(self.gravity_estimates[-50:], axis=0)  # Last 50 estimates
            avg_gravity = avg_gravity / np.linalg.norm(avg_gravity) if np.linalg.norm(avg_gravity) > 1e-6 else avg_gravity
            
        return {
            "total_updates": self.total_updates,
            "accel_used": self.accel_used_count,
            "accel_rejected": self.accel_rejected_count,
            "accel_usage_rate": accel_usage_rate,
            "accel_rejection_rate": accel_rejection_rate,
            "current_bias": self.state.b.copy(),
            "bias_magnitude": np.linalg.norm(self.state.b),
            "avg_gravity_direction": avg_gravity
        }

    # -------------------------------------------------------------------------
    def update(self,
               gyro:   np.ndarray,      # ω   – body frame rad/s
               accel:  np.ndarray,      # a_m – body frame m/s²
               dt:     float) -> Tuple[Quaternion, np.ndarray]:
        """
        One predict/correct step.
        Returns (attitude quaternion, current gyro‑bias estimate).
        """
        s = self.state
        self.total_updates += 1

        # Apply coordinate transforms
        gyro_transformed, accel_transformed = self.apply_coordinate_transforms(gyro, accel)

        # Update separate orientations for compatibility
        self._update_gyro_only_orientation(gyro_transformed, dt)
        self._update_accel_only_orientation(accel_transformed)

        # Initialize main filter orientation from accelerometer on first valid sample
        if not self.is_initialized and np.linalg.norm(accel_transformed) > 1e-6:
            s.q = self.accel_only_orientation  # Initialize to accel-only orientation
            self.is_initialized = True

        # ---------- Normalise accelerometer (â ≈ −g·Rᵀ·e₃) -----------------
        if np.linalg.norm(accel_transformed) < 1e-6:
            a_hat = None            # discard bad sample
            self.accel_rejected_count += 1
        else:
            a_hat = accel_transformed / np.linalg.norm(accel_transformed)
            # Store gravity estimate for debugging
            self.gravity_estimates.append(-a_hat)  # Negative because accel points opposite to gravity
            if len(self.gravity_estimates) > 100:
                self.gravity_estimates = self.gravity_estimates[-100:]

        # ---------- Innovation  ω̃  = vex(π_a(R̃))  ----------------------------
        if a_hat is not None:
            # "Down" in inertial frame is  [0,0,‑1]; predicted down in body frame:
            v_est = s.q.conjugate().rotate_vector(np.array([0, 0, -1]))
            e = np.cross(v_est, -a_hat)            # Eq. (9) vector form
            
            # TARGETED FIX: X-axis appears backwards - flip error sign for X only
            # This compensates for the accelerometer X-axis negation in coordinate transform
            e[0] = -e[0]
            
            self.accel_used_count += 1
        else:
            e     = np.zeros(3)

        # ---------- Bias adaptation  ḃ = −k_i · e  ----------------------------
        if self.k_i > 0.0:
            s.b += -self.k_i * e * dt

        # ---------- Corrected body‑rate  ω_c = ω_m − b + k_p·e  ---------------
        omega_c = gyro_transformed - s.b + self.k_p * e

        # ---------- Attitude integration on SO(3) ------------------------------
        dq      = Quaternion.from_omega(omega_c, dt)   # exp map
        s.q     = s.q * dq
        s.q.normalize()

        return s.q, s.b.copy()

    # --------------------------- Helpers --------------------------------------
    def reset(self):
        self.state = MahonyState()
        self.gyro_only_orientation = Quaternion()
        self.accel_only_orientation = Quaternion()
        self.is_initialized = False
        # Reset debug tracking
        self.total_updates = 0
        self.accel_used_count = 0
        self.accel_rejected_count = 0
        self.gravity_estimates = []
