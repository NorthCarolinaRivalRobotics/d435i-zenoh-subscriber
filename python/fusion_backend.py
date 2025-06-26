import numpy as np
import gtsam
from collections import deque
from gtsam import (Pose3, Point3, Rot3,
                   ISAM2, ISAM2Params, NonlinearFactorGraph, Values,
                   noiseModel, symbol)
from profiling import profiler

class FusionBackend:
    def __init__(self, window_size=5):
        # ISAM2 for sliding window optimization
        p = ISAM2Params()
        p.setFactorization("CHOLESKY")
        self.isam = ISAM2(p)

        # Graph containers
        self.graph = NonlinearFactorGraph()
        self.values = Values()

        # Sliding window management
        self.window_size = window_size
        self.pose_window = deque(maxlen=window_size)
        self.k = 0
        
        # State tracking
        self.current_rotation = np.eye(3)  # Track rotation from gyro integration
        self.current_position = np.zeros(3)  # Track position from camera
        
        # Timing
        self.last_gyro_time = None
        
        self._initialize_first_pose()

    def _initialize_first_pose(self):
        """Initialize the first pose with strong priors."""
        pose0 = Pose3(Rot3(), Point3(0, 0, 0))
        
        # Strong prior for the first pose
        self.graph.add(gtsam.PriorFactorPose3(symbol('x', 0),
                                              pose0,
                                              noiseModel.Diagonal.Variances([1e-6]*6)))
        
        self.values.insert(symbol('x', 0), pose0)
        self.pose_window.append(0)
        
        # Update ISAM2
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)
        self.values.clear()

    def integrate_gyro(self, gyro, timestamp):
        """Integrate gyroscope data to update rotation."""
        with profiler.timer("backend.gyro_integration"):
            if self.last_gyro_time is not None:
                # Convert timestamp from milliseconds to seconds for dt calculation
                dt = (timestamp - self.last_gyro_time) / 1000.0
                if 0.001 < dt < 0.1:  # Reasonable dt bounds (1ms to 100ms)
                    # Angular velocity integration (simple Euler integration)
                    angle_delta = gyro * dt
                    angle_magnitude = np.linalg.norm(angle_delta)
                    
                    if angle_magnitude > 1e-8:
                        # Create rotation matrix from axis-angle
                        axis = angle_delta / angle_magnitude
                        rotation_delta = self._axis_angle_to_rotation_matrix(axis, angle_magnitude)
                        self.current_rotation = self.current_rotation @ rotation_delta
                        
                        # Debug output for significant rotations
                        if angle_magnitude > 0.01:  # > ~0.6 degrees
                            print(f"Gyro integration: dt={dt:.4f}s, angle_delta={np.degrees(angle_magnitude):.2f}deg")
                            
                            # Log current rotation as Euler angles for debugging
                            from scipy.spatial.transform import Rotation as R
                            euler = R.from_matrix(self.current_rotation).as_euler('xyz', degrees=True)
                            print(f"Current rotation (RPY): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg")
                else:
                    print(f"Warning: Invalid gyro dt={dt:.6f}s, skipping integration")

            self.last_gyro_time = timestamp

    def _axis_angle_to_rotation_matrix(self, axis, angle):
        """Convert axis-angle to rotation matrix using Rodrigues' formula."""
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def add_camera_translation(self, translation, translation_cov=None):
        """Add camera translation measurement and optimize sliding window."""
        with profiler.timer("backend.add_translation_total"):
            # Update position from camera
            self.current_position += translation
            
            # Create new pose from integrated rotation and camera position
            new_pose = Pose3(Rot3(self.current_rotation), 
                            Point3(self.current_position[0], 
                                   self.current_position[1], 
                                   self.current_position[2]))
            
            # Increment pose counter for new pose
            kp = self.k + 1
            
            # Add to sliding window
            self.pose_window.append(kp)
            
            # Remove old poses if window is full
            if len(self.pose_window) > self.window_size:
                old_pose_idx = self.pose_window.popleft()
                # Note: In a full implementation, we'd marginalize out old variables
                # For simplicity, we'll just keep recent poses in the factor graph
            
            # Add odometry factor between consecutive poses
            if self.k >= 0:
                try:
                    with profiler.timer("backend.factor_creation"):
                        # Get previous pose
                        prev_pose = self.isam.calculateEstimatePose3(symbol('x', self.k))
                        
                        # Calculate relative transformation
                        relative_transform = prev_pose.inverse().compose(new_pose)
                        
                        # Default covariance if not provided
                        if translation_cov is None:
                            # Higher uncertainty for rotation (gyro drift), lower for translation (camera)
                            cov_diag = [0.1, 0.1, 0.1, 0.05, 0.01, 0.01]  # [rot, rot, rot, trans, trans, trans]
                        else:
                            # Use provided translation covariance and add rotation uncertainty
                            cov_diag = [0.1, 0.1, 0.1] + list(translation_cov)
                        
                        noise = noiseModel.Diagonal.Variances(cov_diag)
                        
                        # Add between factor
                        self.graph.add(gtsam.BetweenFactorPose3(symbol('x', self.k),
                                                                symbol('x', kp),
                                                                relative_transform, 
                                                                noise))
                    
                except Exception as e:
                    print(f"Error creating odometry factor: {e}")
                    # Remove the failed pose from window and don't increment counter
                    if self.pose_window and self.pose_window[-1] == kp:
                        self.pose_window.pop()
                    return False
            
            # Add initial guess for new pose
            self.values.insert(symbol('x', kp), new_pose)
            
            # Update ISAM2
            try:
                with profiler.timer("backend.gtsam_optimization"):
                    self.isam.update(self.graph, self.values)
                    self.isam.update()  # Relinearize
                
                # Clear temporary containers
                self.graph.resize(0)
                self.values.clear()
                
                # Only increment k after successful optimization
                self.k = kp
                print(f"✓ Successfully added pose x{kp}")
                return True
                
            except Exception as e:
                print(f"Error in GTSAM optimization: {e}")
                # Clear failed state - don't increment k
                self.graph.resize(0)
                self.values.clear()
                # Remove the failed pose from window
                if self.pose_window and self.pose_window[-1] == kp:
                    self.pose_window.pop()
                return False

    def get_current_pose(self):
        """Get the current optimized pose."""
        try:
            if self.k >= 0:
                pose = self.isam.calculateEstimatePose3(symbol('x', self.k))
                position = pose.translation()
                rotation = pose.rotation()
                
                # Convert to numpy arrays - GTSAM Point3 has .x(), .y(), .z() methods
                if hasattr(position, 'x'):
                    pos_array = np.array([position.x(), position.y(), position.z()])
                else:
                    # If it's already a numpy array, use it directly
                    pos_array = np.array(position) if not isinstance(position, np.ndarray) else position
                
                if hasattr(rotation, 'toQuaternion'):
                    quat = rotation.toQuaternion()
                    quat_array = np.array([quat.w(), quat.x(), quat.y(), quat.z()])
                else:
                    # Fallback to identity quaternion
                    quat_array = np.array([1.0, 0.0, 0.0, 0.0])
                
                return pos_array, quat_array
            else:
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                
        except Exception as e:
            print(f"Error getting current pose: {e}")
            # Return identity pose as fallback
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def get_trajectory(self):
        """Get the full optimized trajectory within the sliding window."""
        trajectory = []
        try:
            for pose_idx in self.pose_window:
                pose = self.isam.calculateEstimatePose3(symbol('x', pose_idx))
                position = pose.translation()
                rotation = pose.rotation()
                
                pos_array = np.array([position.x(), position.y(), position.z()])
                quat = rotation.toQuaternion()
                quat_array = np.array([quat.w(), quat.x(), quat.y(), quat.z()])
                
                trajectory.append((pos_array, quat_array))
                
        except Exception as e:
            print(f"Error getting trajectory: {e}")
            
        return trajectory

