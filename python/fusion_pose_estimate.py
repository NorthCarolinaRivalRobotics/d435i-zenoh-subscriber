#!/usr/bin/env python3
"""
Fusion-based pose estimation combining IMU and RGB-D visual odometry.
Uses GTSAM factor graph optimization for robust sensor fusion.
"""
from __future__ import annotations

import time
import numpy as np
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from collections import deque
import threading
from abc import ABC, abstractmethod
import rerun as rr

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports
from fusion_backend import FusionBackend
from camera_data_manager import setup_camera_manager
from camera_config import CameraCalibration
from vision_utils import CAMERA_TO_ROBOT_FRAME, estimate_multiframe_transform
from imu_utils import Quaternion, load_calibration, IMUProcessor, create_axis_swap_matrix
from visualization import RerunVisualizer
from profiling import profiler

# Add parent directory for zenoh_d435i_subscriber
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    import zenoh_d435i_subscriber as zd435i
except ImportError as e:
    print(f"Error importing zenoh_d435i_subscriber: {e}")
    print("Make sure the module is compiled with: maturin develop")
    sys.exit(1)


@dataclass
class SensorMeasurement:
    """Base class for sensor measurements."""
    timestamp: float
    frame_id: int


@dataclass
class IMUMeasurement(SensorMeasurement):
    """IMU measurement data."""
    gyro: np.ndarray  # rad/s
    accel: np.ndarray  # m/s^2
    dt: float  # Time delta since last measurement


@dataclass
class VisualOdometryMeasurement(SensorMeasurement):
    """Visual odometry measurement data."""
    transform: np.ndarray  # 4x4 SE(3) transform
    covariance: np.ndarray  # 6x6 covariance matrix (rot, trans)
    num_inliers: int
    points_prev: np.ndarray  # 3D points in previous frame
    points_curr: np.ndarray  # 3D points in current frame
    keypoints_prev: list = field(default_factory=list)  # Keypoints from first frame
    keypoints_curr: list = field(default_factory=list)  # Keypoints from last frame  
    matches: list = field(default_factory=list)  # Matches between frames
    rgb_prev: np.ndarray = None  # RGB image from first frame (optional)
    rgb_curr: np.ndarray = None  # RGB image from last frame (optional)


class SensorInterface(ABC):
    """Abstract interface for sensor data sources."""
    
    @abstractmethod
    def get_measurement(self) -> Optional[SensorMeasurement]:
        """Get latest measurement from sensor."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if sensor is ready to provide data."""
        pass


class GyroProcessor:
    """Simple gyroscope data processor for rotation tracking."""
    
    def __init__(self, coordinate_transform=None):
        self.coordinate_transform = coordinate_transform if coordinate_transform is not None else np.eye(3)
        
    def process_gyro(self, raw_gyro):
        """Process raw gyroscope data."""
        # Apply coordinate transformation
        transformed_gyro = self.coordinate_transform @ raw_gyro
        return transformed_gyro


class GyroSensor:
    """Gyroscope sensor interface."""
    
    def __init__(self, subscriber: zd435i.ZenohD435iSubscriber, coordinate_transform=None):
        self.subscriber = subscriber
        self.processor = GyroProcessor(coordinate_transform)
        self.last_frame_count = -1
        
    def get_gyro_measurement(self):
        """Get processed gyroscope measurement."""
        frame_data = self.subscriber.get_latest_frames()
        
        if frame_data.frame_count == 0 or frame_data.motion is None:
            return None
            
        # Skip if same frame as before
        if frame_data.frame_count == self.last_frame_count:
            return None
            
        self.last_frame_count = frame_data.frame_count
        
        # Extract gyroscope data
        gyro_raw = np.array([
            frame_data.motion.gyro[0],
            frame_data.motion.gyro[1], 
            frame_data.motion.gyro[2]
        ])
        
        # Process the data
        gyro_processed = self.processor.process_gyro(gyro_raw)
        
        # Debug output for significant gyro values
        raw_mag = np.linalg.norm(gyro_raw)
        processed_mag = np.linalg.norm(gyro_processed)
        if raw_mag > 0.05 or processed_mag > 0.05:  # > ~3 degrees/sec
            print(f"Gyro sensor: raw_mag={raw_mag:.4f}, processed_mag={processed_mag:.4f} rad/s")
            print(f"  Raw: [{gyro_raw[0]:+.4f}, {gyro_raw[1]:+.4f}, {gyro_raw[2]:+.4f}]")
            print(f"  Processed: [{gyro_processed[0]:+.4f}, {gyro_processed[1]:+.4f}, {gyro_processed[2]:+.4f}]")
        
        return {
            'gyro': gyro_processed,
            'timestamp': frame_data.motion.timestamp,
            'frame_id': frame_data.frame_count
        }


class IMUSensor(SensorInterface):
    """IMU sensor interface with calibration and coordinate transforms."""
    
    def __init__(self, subscriber: zd435i.ZenohD435iSubscriber, 
                 calibration: Optional[Dict] = None,
                 gyro_transform: Optional[np.ndarray] = None):
        self.subscriber = subscriber
        self.processor = IMUProcessor(calibration)
        self.gyro_transform = gyro_transform if gyro_transform is not None else np.eye(3)
        self.last_timestamp = None
        self.last_frame_count = -1  # Track last processed frame ID
        self.frame_id = 0
        
    def get_measurement(self) -> Optional[IMUMeasurement]:
        """Get calibrated IMU measurement."""
        with profiler.timer("IMU.get_measurement"):
            frame_data = self.subscriber.get_latest_frames()
            
            if frame_data.frame_count == 0 or frame_data.motion is None:
                return None
                
            # Check if this is a new frame - crucial for avoiding duplicate processing!
            if frame_data.frame_count == self.last_frame_count:
                return None
                
            # Update frame tracking
            self.last_frame_count = frame_data.frame_count
                
            # Extract raw data
            raw_gyro = np.array(frame_data.motion.gyro)
            raw_accel = np.array(frame_data.motion.accel)
            timestamp = frame_data.motion.timestamp
            
            # Apply calibration
            with profiler.timer("IMU.calibration"):
                cal_gyro, cal_accel = self.processor.process_sample(raw_gyro, raw_accel)
            
            # Apply coordinate transform
            cal_gyro = self.gyro_transform @ cal_gyro
            cal_accel = self.gyro_transform @ cal_accel
            
            # Calculate dt with validation
            dt = 0.01  # Default 10ms (100Hz)
            if self.last_timestamp is not None:
                calculated_dt = (timestamp - self.last_timestamp) / 1000.0  # ms to seconds
                # Validate dt: must be positive and reasonable (between 1ms and 100ms)
                if 0.001 <= calculated_dt <= 0.1:  
                    dt = calculated_dt
                else:
                    # Log warning for debugging but continue with default dt
                    if calculated_dt <= 0:
                        print(f"Warning: Invalid dt={calculated_dt:.6f}s (timestamp went backwards or same), using default")
                    else:
                        print(f"Warning: Unreasonable dt={calculated_dt:.3f}s, using default")
            
            self.last_timestamp = timestamp
            self.frame_id += 1
            
            return IMUMeasurement(
                timestamp=timestamp,
                frame_id=self.frame_id,
                gyro=cal_gyro,
                accel=cal_accel,
                dt=dt
            )
    
    def is_ready(self) -> bool:
        """Check if IMU data is available."""
        frame_data = self.subscriber.get_latest_frames()
        return frame_data.frame_count > 0 and frame_data.motion is not None


class VisualOdometrySensor(SensorInterface):
    """Visual odometry sensor using multi-frame RGB-D tracking."""
    
    def __init__(self, camera_manager, camera_cal: CameraCalibration, 
                 n_frames: int = 4, min_inliers: int = 10):
        self.camera_manager = camera_manager
        self.camera_cal = camera_cal
        self.n_frames = n_frames
        self.min_inliers = min_inliers
        self.frame_buffer = deque(maxlen=n_frames)
        self.last_frame_id = -1
        self.frame_id = 0
        
    def get_measurement(self) -> Optional[VisualOdometryMeasurement]:
        """Get visual odometry measurement from multi-frame tracking."""
        with profiler.timer("VO.get_measurement"):
            fd = self.camera_manager.get_latest_frames()
            
            if fd.frame_count == 0:
                return None
                
            # Check for new frame
            if fd.frame_count == self.last_frame_id:
                return None
                
            self.last_frame_id = fd.frame_count
            
            try:
                # Decode images
                with profiler.timer("VO.decode_images"):
                    rgb_buf = fd.rgb.get_data()
                    w, h = fd.rgb.width, fd.rgb.height
                    
                    # Debug RGB buffer
                    print(f"[DEBUG] RGB buffer: size={len(rgb_buf)} bytes, expected dimensions={w}x{h}")
                    if len(rgb_buf) < 100:
                        print(f"[DEBUG] RGB buffer too small! First 50 bytes: {rgb_buf[:50].hex() if rgb_buf else 'EMPTY'}")
                    
                    import cv2
                    
                    # Check if buffer is raw RGB based on size
                    expected_raw_size = w * h * 3
                    if len(rgb_buf) == expected_raw_size:
                        # Raw RGB data
                        rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))
                    else:
                        # Try JPEG decode
                        rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                        if rgb_bgr is None:
                            print(f"[DEBUG] JPEG decode failed for buffer size {len(rgb_buf)}")
                    
                    depth_img = fd.depth.get_data_2d().astype(np.float32)
                
                # Add to buffer
                timestamp = time.time() * 1000  # Convert to ms
                # Convert BGR to RGB before storing
                if rgb_bgr is not None:
                    rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB) 
                else:
                    rgb_img = None
                    
                # Only add to buffer if we have valid RGB data
                if rgb_img is not None:
                    self.frame_buffer.append({
                        "rgb": rgb_img,
                        "depth": depth_img,
                        "id": fd.frame_count,
                        "timestamp": timestamp
                    })
                    
                    # Debug: Log frame buffer status
                    print(f"[VO DEBUG] Frame buffer: {len(self.frame_buffer)}/{self.n_frames} frames, "
                          f"frame_id={fd.frame_count}, timestamp={timestamp:.1f}ms")
                else:
                    print(f"[VO DEBUG] Skipping frame {fd.frame_count} - no valid RGB data")
                
                # Need full buffer for multi-frame tracking
                if len(self.frame_buffer) < self.n_frames:
                    print(f"[VO DEBUG] Waiting for more frames... ({len(self.frame_buffer)}/{self.n_frames})")
                    return None
                    
                # Perform multi-frame tracking
                frames_data = [(frame["rgb"], frame["depth"]) for frame in self.frame_buffer]
                print(f"Frames data: {len(frames_data)}")
                
                with profiler.timer("VO.estimate_transform"):
                    result = estimate_multiframe_transform(
                        frames_data,
                        self.camera_cal.K_rgb, 
                        self.camera_cal.K_depth,
                        self.camera_cal.T_rgb_to_depth,
                        depth_scale=1.0,
                        return_keypoints_matches=True
                    )
                
                # Unpack results based on return_keypoints_matches flag
                if len(result) == 7:
                    P1, P2, T, num_tracks, kp_prev, kp_curr, matches = result
                else:
                    P1, P2, T, num_tracks = result
                    kp_prev, kp_curr, matches = [], [], []
                
                if T is None or num_tracks < self.min_inliers:
                    return None
                    
                # Estimate covariance based on number of inliers and reprojection error
                # Simple heuristic: fewer inliers = higher uncertainty
                base_cov = np.eye(6) * 0.01  # Base covariance
                cov_scale = max(1.0, 50.0 / num_tracks)  # Scale inversely with inliers
                covariance = base_cov * cov_scale
                
                self.frame_id += 1
                
                return VisualOdometryMeasurement(
                    timestamp=self.frame_buffer[-1]["timestamp"],
                    frame_id=self.frame_id,
                    transform=T,
                    covariance=covariance,
                    num_inliers=num_tracks,
                    points_prev=P1,
                    points_curr=P2,
                    keypoints_prev=kp_prev,
                    keypoints_curr=kp_curr,
                    matches=matches,
                    rgb_prev=self.frame_buffer[0]["rgb"],  # First frame
                    rgb_curr=self.frame_buffer[-1]["rgb"]  # Last frame
                )
                
            except Exception as e:
                print(f"Error in visual odometry: {e}")
                return None
    
    def is_ready(self) -> bool:
        """Check if enough frames are buffered."""
        return len(self.frame_buffer) >= self.n_frames


class FusionVisualizer(RerunVisualizer):
    """Extended visualizer for sensor fusion debugging."""
    
    def __init__(self, app_name: str = "fusion_pose_estimate", spawn: bool = True,
                 plot_3d_matches: bool = False):
        super().__init__(app_name, spawn)
        self.plot_3d_matches = plot_3d_matches
        self.setup_fusion_layout()
        
    def setup_fusion_layout(self):
        """Set up Rerun layout for fusion visualization."""
        # Log static coordinate frames with static=True to avoid re-logging
        self.log_coordinate_frame("world", scale=1.0, static=True)
        self.log_coordinate_frame("imu", scale=0.3, color=[255, 0, 0], static=True)
        self.log_coordinate_frame("camera", scale=0.3, color=[0, 255, 0], static=True)
        
    def log_coordinate_frame(self, name: str, scale: float = 1.0, 
                           color: list[int] = None, transform: np.ndarray = None,
                           static: bool = False):
        """Log a coordinate frame with axes."""
        origins = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        vectors = [[scale, 0, 0], [0, scale, 0], [0, 0, scale]]
        
        if transform is not None:
            # Transform the vectors
            R = transform[:3, :3]
            t = transform[:3, 3]
            origins = [t, t, t]
            vectors = [R @ v for v in vectors]
        
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
        if color is not None:
            # Blend the provided color with each axis color
            # Simple approach: use the provided color for all axes but keep some distinction
            base_color = np.array(color)
            colors = [
                [int(base_color[0] * 1.0), int(base_color[1] * 0.3), int(base_color[2] * 0.3)],  # X: more red component
                [int(base_color[0] * 0.3), int(base_color[1] * 1.0), int(base_color[2] * 0.3)],  # Y: more green component  
                [int(base_color[0] * 0.3), int(base_color[1] * 0.3), int(base_color[2] * 1.0)]   # Z: more blue component
            ]
            # Clamp values to 0-255 range
            colors = [[max(0, min(255, c)) for c in col] for col in colors]
            
        rr.log(f"{name}/axes", rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=colors,
            labels=["X", "Y", "Z"]
        ), static=static)
    
    def log_imu_data(self, measurement: IMUMeasurement):
        """Log IMU measurement data."""
        import rerun as rr
        
        # Don't set time here - it's set in main loop
        # rr.set_time_seconds("timestamp", measurement.timestamp / 1000.0)
        
        # Initialize counter for reduced frequency logging
        if not hasattr(self, '_imu_log_counter'):
            self._imu_log_counter = 0
        self._imu_log_counter += 1
        
        # Only log scalars every 5th IMU measurement to reduce load
        if self._imu_log_counter % 5 == 0:
            # Log raw values
            rr.log("imu/gyro_x", rr.Scalar(measurement.gyro[0]))
            rr.log("imu/gyro_y", rr.Scalar(measurement.gyro[1]))
            rr.log("imu/gyro_z", rr.Scalar(measurement.gyro[2]))
            rr.log("imu/gyro_magnitude", rr.Scalar(np.linalg.norm(measurement.gyro)))
            
            rr.log("imu/accel_x", rr.Scalar(measurement.accel[0]))
            rr.log("imu/accel_y", rr.Scalar(measurement.accel[1]))
            rr.log("imu/accel_z", rr.Scalar(measurement.accel[2]))
            rr.log("imu/accel_magnitude", rr.Scalar(np.linalg.norm(measurement.accel)))
    
    def log_visual_odometry_data(self, measurement: VisualOdometryMeasurement):
        """Log visual odometry measurement."""
        import rerun as rr
        
        # Don't set time here - it's set in main loop
        # rr.set_time_seconds("timestamp", measurement.timestamp / 1000.0)
        
        # Log transform components
        translation = measurement.transform[:3, 3]
        rr.log("vo/translation_x", rr.Scalar(translation[0]))
        rr.log("vo/translation_y", rr.Scalar(translation[1]))
        rr.log("vo/translation_z", rr.Scalar(translation[2]))
        rr.log("vo/translation_magnitude", rr.Scalar(np.linalg.norm(translation)))
        
        # Log quality metrics
        rr.log("vo/num_inliers", rr.Scalar(measurement.num_inliers))
        rr.log("vo/covariance_trace", rr.Scalar(np.trace(measurement.covariance)))
        
        # Log 3D matches only if enabled
        if self.plot_3d_matches and len(measurement.points_prev) > 0:
            self.log_3d_matches(measurement.points_prev, measurement.points_curr)
            
        # Log 2D feature matches if available - reduce frequency
        if not hasattr(self, '_vo_match_log_counter'):
            self._vo_match_log_counter = 0
        self._vo_match_log_counter += 1
        
        # Only log matches every 3rd frame to reduce load
        if len(measurement.keypoints_prev) > 0 and len(measurement.keypoints_curr) > 0 and len(measurement.matches) > 0 and measurement.rgb_prev is not None and measurement.rgb_curr is not None:
            self.log_2d_feature_matches(
                measurement.keypoints_prev,
                measurement.keypoints_curr,
                measurement.matches,
                measurement.rgb_prev,
                measurement.rgb_curr
            )
    
    def log_fusion_state(self, pose: Tuple[np.ndarray, np.ndarray], 
                        velocity: np.ndarray = None, bias: Dict = None):
        """Log fused state estimate."""
        import rerun as rr
        
        position, quaternion = pose
        
        # Log fused position
        rr.log("fusion/position", rr.Points3D([position], colors=[255, 255, 0], radii=0.05))
        
        # Don't log trajectory point here - we'll handle it in main loop
        # rr.log("fusion/trajectory", rr.Points3D([position], colors=[255, 200, 0], radii=0.02))
        
        # Create transform from quaternion
        q = Quaternion.from_array(quaternion)
        R = q.to_rotation_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        # Log coordinate frame only occasionally (removed to reduce overhead)
        # self.log_coordinate_frame("fusion/current", scale=1.0, color=[255, 255, 0], transform=T)
        
        # Log a single orientation arrow instead of multiple
        forward_vector = R @ np.array([1.0, 0, 0])  # Forward is +X direction
        rr.log("fusion/orientation", rr.Arrows3D(
            origins=[position],
            vectors=[forward_vector * 0.8],  # Scale the arrow
            colors=[255, 255, 0],  # Yellow for visibility
            radii=0.02
        ))
        
        # Remove redundant orientation vectors
        # up_vector = R @ np.array([0, 0, 1.0])  # Up is +Z direction
        # right_vector = R @ np.array([0, 1.0, 0])  # Right is +Y direction
        
        # Log state values (reduce frequency of scalar logging)
        # Only log every 10th call to reduce load
        if not hasattr(self, '_scalar_log_counter'):
            self._scalar_log_counter = 0
        self._scalar_log_counter += 1
        
        if self._scalar_log_counter % 10 == 0:
            rr.log("fusion/position_x", rr.Scalar(position[0]))
            rr.log("fusion/position_y", rr.Scalar(position[1]))
            rr.log("fusion/position_z", rr.Scalar(position[2]))
            
            # Log orientation as Euler angles for easier interpretation
            from scipy.spatial.transform import Rotation as scipy_R
            euler = scipy_R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_euler('xyz', degrees=True)
            rr.log("fusion/roll_deg", rr.Scalar(euler[0]))
            rr.log("fusion/pitch_deg", rr.Scalar(euler[1]))
            rr.log("fusion/yaw_deg", rr.Scalar(euler[2]))
            
            # Log quaternion components
            rr.log("fusion/quat_w", rr.Scalar(quaternion[0]))
            rr.log("fusion/quat_x", rr.Scalar(quaternion[1]))
            rr.log("fusion/quat_y", rr.Scalar(quaternion[2]))
            rr.log("fusion/quat_z", rr.Scalar(quaternion[3]))
        
        if velocity is not None:
            rr.log("fusion/velocity_magnitude", rr.Scalar(np.linalg.norm(velocity)))
            
        if bias is not None and "gyro" in bias:
            rr.log("fusion/gyro_bias_magnitude", rr.Scalar(np.linalg.norm(bias["gyro"])))


class FusionPoseEstimator:
    """Simplified fusion pose estimator using gyro rotation + camera translation."""
    
    @staticmethod
    def create_axis_transform(x_from='X', y_from='Y', z_from='Z', 
                            x_sign=1, y_sign=1, z_sign=1):
        """Create axis transformation matrix with optional sign inversions.
        
        Args:
            x_from: Which IMU axis maps to robot X ('X', 'Y', 'Z')
            y_from: Which IMU axis maps to robot Y ('X', 'Y', 'Z')
            z_from: Which IMU axis maps to robot Z ('X', 'Y', 'Z')
            x_sign: Sign for X mapping (1 or -1)
            y_sign: Sign for Y mapping (1 or -1)
            z_sign: Sign for Z mapping (1 or -1)
        
        Returns:
            3x3 transformation matrix
        """
        transform = np.zeros((3, 3))
        
        # Map axes
        axis_map = {'X': 0, 'Y': 1, 'Z': 2}
        transform[0, axis_map[x_from]] = x_sign
        transform[1, axis_map[y_from]] = y_sign
        transform[2, axis_map[z_from]] = z_sign
        
        return transform
    
    def __init__(self, window_size=5):
        self.backend = FusionBackend(window_size=window_size)
        self.gyro_sensor = None
        self.vo_system = None
        
        # Threading
        self.running = False
        self.fusion_thread = None
        self.queue_lock = threading.Lock()
        
        # Measurement queues
        self.gyro_queue = deque(maxlen=1000)
        self.translation_queue = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'gyro_measurements': 0,
            'translation_measurements': 0,
            'fusion_updates': 0,
            'last_update_time': time.time()
        }
        
    def initialize_sensors(self, subscriber, camera_manager, camera_cal):
        """Initialize sensor interfaces."""
        print("Initializing sensors...")
        
        # Initialize gyroscope sensor with coordinate transform
        # We need to transform from camera/IMU frame to robot frame
        # Camera frame (pinhole model): Z=forward, X=right, Y=down
        # Robot frame: X=forward, Y=right, Z=up
        # 
        # Based on observations:
        # - Yaw rotates around camera X (should be robot Z)
        # - Forward motion is along camera Z (should be robot X)
        # - X axis needs negation, Y is good as is, Z should NOT be negated
        # 
        # Transform mapping:
        # Robot X (forward) ← Camera -Z (negated)
        # Robot Y (right) ← Camera X
        # Robot Z (up) ← Camera Y (no negation needed)
        
        gyro_transform = self.create_axis_transform(
            x_from='Z',   # Robot X ← Camera Z
            y_from='X',   # Robot Y ← Camera X
            z_from='Y',   # Robot Z ← Camera Y
            x_sign=-1,    # Negate X mapping (Camera -Z → Robot X)
            z_sign=-1      # No negation for Z mapping (Camera Y → Robot Z)
        )
        
        print("Gyro coordinate transform (Camera to Robot):")
        print(gyro_transform)
        print("This maps: Camera[X=right,Y=down,Z=forward] → Robot[X=forward,Y=right,Z=up]")
        print("With mappings: Camera -Z → Robot X, Camera X → Robot Y, Camera Y → Robot Z")
        
        self.gyro_sensor = GyroSensor(subscriber, gyro_transform)
        
        # Store camera-to-robot transform for visual odometry
        self.camera_to_robot_transform = gyro_transform
        
        # Initialize visual odometry system
        self.vo_system = VisualOdometrySensor(camera_manager, camera_cal, n_frames=3)
        
        
    def sensor_data_thread(self):
        """Background thread to collect sensor data."""
        while self.running:
            with profiler.timer("sensor_data_thread"):
                # Collect gyroscope data
                gyro_measurement = self.gyro_sensor.get_gyro_measurement()
                if gyro_measurement is not None:
                    with self.queue_lock:
                        self.gyro_queue.append(gyro_measurement)
                        self.stats['gyro_measurements'] += 1
                
                # Collect visual odometry data  
                vo_measurement = self.vo_system.get_measurement()
                if vo_measurement is not None:
                    # Extract translation and transform from camera to robot frame
                    translation_camera = vo_measurement.transform[:3, 3]
                    translation_robot = self.camera_to_robot_transform @ translation_camera
                    
                    translation_data = {
                        'translation': translation_robot,
                        'covariance': np.diag(vo_measurement.covariance)[3:6],  # Translation part
                        'timestamp': time.time()
                    }
                    
                    with self.queue_lock:
                        self.translation_queue.append(translation_data)
                        self.stats['translation_measurements'] += 1
                
                time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def run_fusion_step(self):
        """Run one fusion step."""
        with profiler.timer("fusion_step"):
            # Process all available gyro measurements
            gyro_measurements_processed = 0
            with self.queue_lock:
                while self.gyro_queue:
                    gyro_data = self.gyro_queue.popleft()
                    # Add debug output
                    gyro_magnitude = np.linalg.norm(gyro_data['gyro'])
                    if gyro_magnitude > 0.01:  # Only print for significant rotation
                        print(f"Processing gyro: magnitude={gyro_magnitude:.4f} rad/s, timestamp={gyro_data['timestamp']}")
                    
                    # Integrate gyroscope data
                    with profiler.timer("backend.integrate_gyro"):
                        self.backend.integrate_gyro(gyro_data['gyro'], gyro_data['timestamp'])
                    gyro_measurements_processed += 1
            
            # Process camera translation measurements
            translation_processed = False
            with self.queue_lock:
                if self.translation_queue:
                    translation_data = self.translation_queue.popleft()
                    # Add debug output
                    trans_magnitude = np.linalg.norm(translation_data['translation'])
                    # Add camera translation measurement
                    with profiler.timer("backend.add_camera_translation"):
                        success = self.backend.add_camera_translation(
                            translation_data['translation'],
                            translation_data['covariance']
                        )
                    if success:
                        translation_processed = True
                        self.stats['fusion_updates'] += 1
            
            return gyro_measurements_processed > 0 or translation_processed
    
    def get_current_state(self):
        """Get current pose estimate."""
        return self.backend.get_current_pose()
    
    def get_statistics(self):
        """Get system statistics."""
        current_time = time.time()
        dt = current_time - self.stats['last_update_time']
        
        if dt > 0:
            gyro_rate = self.stats['gyro_measurements'] / dt
            translation_rate = self.stats['translation_measurements'] / dt
        else:
            gyro_rate = 0.0
            translation_rate = 0.0
        
        return {
            'gyro_measurements': self.stats['gyro_measurements'],
            'translation_measurements': self.stats['translation_measurements'], 
            'fusion_updates': self.stats['fusion_updates'],
            'gyro_rate': gyro_rate,
            'translation_rate': translation_rate
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'gyro_measurements': 0,
            'translation_measurements': 0,
            'fusion_updates': 0,
            'last_update_time': time.time()
        }
    
    def start(self):
        """Start the fusion system."""
        self.running = True
        self.fusion_thread = threading.Thread(target=self.sensor_data_thread)
        self.fusion_thread.start()
    
    def stop(self):
        """Stop the fusion system."""
        self.running = False
        if self.fusion_thread:
            self.fusion_thread.join()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simplified Fusion-based Pose Estimation")
    parser.add_argument('mode', nargs='?', default='LIVE', 
                        help='Camera mode (LIVE or other)')
    parser.add_argument('--plot-3d-matches', action='store_true',
                        help='Enable plotting of 3D point matches (impacts performance)')
    args = parser.parse_args()
    
    # Initialize Rerun with proper visualizer
    print("=== Simplified Fusion-based Pose Estimation ===")
    print("Gyro rotation + Camera translation with sliding window optimization")
    print()
    
    # Create the fusion visualizer (this will initialize Rerun and spawn the viewer)
    visualizer = FusionVisualizer("fusion_pose_estimation", spawn=True, 
                                 plot_3d_matches=args.plot_3d_matches)
    
    if args.plot_3d_matches:
        print("✓ 3D match visualization ENABLED (may impact performance)")
    else:
        print("✓ 3D match visualization DISABLED (use --plot-3d-matches to enable)")
    
    # Set up camera manager
    camera_manager, cam_args = setup_camera_manager("Simplified fusion pose estimation")
    camera_cal = CameraCalibration.create_default_d435i()
    
    # Get camera mode
    camera_mode = args.mode.upper()
    
    print(f"Camera mode: [{camera_mode}]")
    
    # Initialize subscriber
    subscriber = zd435i.ZenohD435iSubscriber()
    
    # Connect to data sources
    try:
        print("Connecting to Zenoh...")
        subscriber.connect()
        subscriber.start_subscribing()
        print("✓ Connected to IMU data")
        
        print("Starting camera manager...")
        camera_manager.connect()
        camera_manager.start_subscribing()
        print("✓ Connected to camera data")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
    
    # Set up coordinate transforms
    print("✓ Set up coordinate transforms")
    
    # Initialize fusion system
    fusion_estimator = FusionPoseEstimator(window_size=2)
    fusion_estimator.initialize_sensors(subscriber, camera_manager, camera_cal)
    
    # Wait for initial sensor data
    print("Waiting for sensor data...")
    start_wait_time = time.time()
    while not (camera_manager.is_running()):
        time.sleep(0.1)
        elapsed = time.time() - start_wait_time
        if int(elapsed) % 1 == 0 and elapsed - int(elapsed) < 0.1:  # Print every second
            print(f"  Still waiting for camera manager... ({elapsed:.1f}s)")
    
    print(f"Camera manager ready after {time.time() - start_wait_time:.1f}s")
    print("Waiting additional 2 seconds for sensor stabilization...")
    time.sleep(2.0)
    print("✓ Sensors ready")
    print()
    
    # Start fusion
    print("Starting simplified sensor fusion...")
    print("Press Ctrl+C to stop")
    print()
    
    fusion_estimator.start()
    
    try:
        last_report_time = time.time()
        trajectory_points = []
        last_camera_frame_id = -1  # Track last logged camera frame
        camera_frame_skip_counter = 0  # Skip some camera frames to reduce load
        camera_frames_logged = 0  # Track how many frames we actually log
        
        # Frame rate tracking
        frame_timestamps = deque(maxlen=30)  # Track last 30 frames for FPS calculation
        last_fps_report = time.time()
        
        while True:
            with profiler.timer("main_loop"):
                # Set time ONCE at the beginning of each loop iteration
                current_time = time.time()
                rr.set_time_seconds("timestamp", current_time)
                
                # Get current sensor measurements for visualization
                gyro_measurement = fusion_estimator.gyro_sensor.get_gyro_measurement()
                if gyro_measurement is not None:
                    # Convert to IMUMeasurement format for visualization
                    imu_meas = IMUMeasurement(
                        timestamp=gyro_measurement['timestamp'],
                        frame_id=gyro_measurement['frame_id'],
                        gyro=gyro_measurement['gyro'],
                        accel=np.zeros(3),  # We don't use accelerometer in this version
                        dt=0.01
                    )
                    with profiler.timer("visualizer.log_imu"):
                        visualizer.log_imu_data(imu_meas)
                
                # Get visual odometry measurement for visualization
                vo_measurement = fusion_estimator.vo_system.get_measurement()
                if vo_measurement is not None:
                    with profiler.timer("visualizer.log_vo"):
                        visualizer.log_visual_odometry_data(vo_measurement)
                
                # Run fusion step
                fusion_estimator.run_fusion_step()
                
                # Get current pose and log to visualizer
                with profiler.timer("get_pose"):
                    position, quaternion = fusion_estimator.get_current_state()
                
                # Log current fusion state
                with profiler.timer("visualizer.log_fusion"):
                    visualizer.log_fusion_state((position, quaternion))
                
                # Add to trajectory
                trajectory_points.append(position.copy())
                if len(trajectory_points) > 1000:  # Keep last 1000 points
                    trajectory_points.pop(0)
                
                # Log trajectory less frequently to reduce load
                # Initialize trajectory logging counter if not exists
                if not hasattr(fusion_estimator, '_trajectory_log_counter'):
                    fusion_estimator._trajectory_log_counter = 0
                fusion_estimator._trajectory_log_counter += 1
                
                # Only log trajectory every 10 frames (10Hz if running at 100Hz)
                if len(trajectory_points) > 1 and fusion_estimator._trajectory_log_counter % 10 == 0:
                    with profiler.timer("visualizer.log_trajectory"):
                        rr.log("trajectory", rr.LineStrips3D([trajectory_points], colors=[255, 255, 0]))
                
                # Log camera images when available
                frame_data = camera_manager.get_latest_frames()
                if frame_data.frame_count > 0 and frame_data.frame_count != last_camera_frame_id:
                    # Debug: Log when we get a new camera frame
                    print(f"[DEBUG] New camera frame: ID={frame_data.frame_count}, time={current_time:.6f}")
                    
                    # Track camera frame timestamps for FPS calculation
                    frame_timestamps.append(current_time)
                    if len(frame_timestamps) >= 2:
                        # Calculate instantaneous FPS from last two frames
                        instant_fps = 1.0 / (frame_timestamps[-1] - frame_timestamps[-2])
                        # Calculate average FPS over all tracked frames
                        if len(frame_timestamps) == frame_timestamps.maxlen:
                            avg_fps = len(frame_timestamps) / (frame_timestamps[-1] - frame_timestamps[0])
                            print(f"[DEBUG] Camera FPS: instant={instant_fps:.1f}, avg={avg_fps:.1f}")
                        else:
                            print(f"[DEBUG] Camera FPS: instant={instant_fps:.1f} (building average...)")
                    
                    camera_frame_skip_counter += 1
                    # Only log every 2nd camera frame to reduce load (roughly 15 FPS if camera is 30 FPS)
                    if camera_frame_skip_counter % 2 == 0:
                        print(f"[DEBUG] Logging camera frame (skip_counter={camera_frame_skip_counter})")
                        try:
                            with profiler.timer("visualizer.log_images"):
                                # Decode both images first
                                rgb_buf = frame_data.rgb.get_data()
                                w, h = frame_data.rgb.width, frame_data.rgb.height
                                
                                import cv2
                                
                                # Check if buffer is raw RGB based on size
                                expected_raw_size = w * h * 3
                                if len(rgb_buf) == expected_raw_size:
                                    # Raw RGB data
                                    rgb_bgr = np.frombuffer(rgb_buf, np.uint8).reshape((h, w, 3))
                                else:
                                    # Try JPEG decode
                                    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
                                    if rgb_bgr is None:
                                        print(f"[DEBUG] JPEG decode failed for buffer size {len(rgb_buf)}")
                                
                                depth_img = frame_data.depth.get_data_2d().astype(np.float32)
                                
                                # Only log if BOTH images are valid
                                if rgb_bgr is not None and depth_img is not None:
                                    rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                                    rr.log("camera/rgb", rr.Image(rgb_img))
                                    rr.log("camera/depth", rr.DepthImage(depth_img, meter=1.0))
                                    camera_frames_logged += 1
                                    print(f"[DEBUG] Successfully logged frame at {current_time:.6f}")
                                else:
                                    print(f"[DEBUG] Failed to decode images: rgb={rgb_bgr is not None}, depth={depth_img is not None}")
                            
                            # Update last frame ID after successful logging
                            last_camera_frame_id = frame_data.frame_count
                                    
                        except Exception as e:
                            print(f"[DEBUG] Exception logging images: {e}")
                            pass  # Continue if image logging fails
                    else:
                        # Still update last frame ID even if we skip logging
                        last_camera_frame_id = frame_data.frame_count
                        print(f"[DEBUG] Skipping frame log (counter={camera_frame_skip_counter})")
                elif frame_data.frame_count == 0:
                    # Log once if no frames are available
                    if not hasattr(fusion_estimator, '_no_frames_logged'):
                        print("[DEBUG] No camera frames available yet (frame_count=0)")
                        fusion_estimator._no_frames_logged = True
                
                # Report status every 5 seconds
                if current_time - last_report_time >= 5.0:
                    stats = fusion_estimator.get_statistics()
                    
                    # Convert quaternion to Euler angles for display
                    from scipy.spatial.transform import Rotation as R
                    euler = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_euler('xyz', degrees=True)
                    
                    print(f"\n=== Fusion State (t={current_time:.1f}s) ===")
                    print(f"Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m")
                    print(f"Orientation (RPY): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg")
                    print()
                    print("Sensor Stats:")
                    print(f"  Gyro measurements: {stats['gyro_measurements']}")
                    print(f"  Translation measurements: {stats['translation_measurements']}")
                    print(f"  Fusion updates: {stats['fusion_updates']}")
                    print(f"  Gyro rate: {stats['gyro_rate']:.1f} Hz")
                    print(f"  Translation rate: {stats['translation_rate']:.1f} Hz")
                    print(f"  Camera frames logged: {camera_frames_logged} ({camera_frames_logged/5.0:.1f} Hz)")
                    print("=" * 50)
                    
                    # Reset statistics
                    fusion_estimator.reset_statistics()
                    last_report_time = current_time
                    camera_frames_logged = 0  # Reset camera frame counter
                    
                    # Print profiling results
                    print("\n=== PROFILING RESULTS ===")
                    profiler.print_results(min_calls=10)
                
                # Frame rate tracking
                # frame_timestamps.append(current_time)  # REMOVED: This was incorrectly tracking main loop iterations
                # if current_time - last_fps_report >= 1.0:
                #     fps = len(frame_timestamps) / (current_time - last_fps_report)
                #     print(f"Current frame rate: {fps:.1f} FPS")
                #     last_fps_report = current_time
                
                time.sleep(0.01)  # 100 Hz fusion loop
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        fusion_estimator.stop()
        
    finally:
        # Clean shutdown
        print("Stopping sensors...")
        subscriber.stop()
        camera_manager.stop()
        print("Done!")


if __name__ == "__main__":
    main()
