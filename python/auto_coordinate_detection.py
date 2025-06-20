#!/usr/bin/env python3
"""
Automatic coordinate frame detection for IMU data.
This program analyzes gyroscope and accelerometer responses to motion
to automatically deduce the required coordinate transformations.

The approach:
1. Collect data while user moves the device
2. Extract tilt angles from accelerometer using gravity vector
3. Differentiate tilt angles to get angular rates
4. Compare angular rates from both sensors to find axis alignment
5. Generate the transformation matrices automatically
"""

import time
import numpy as np
import sys
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

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


def extract_roll_pitch_from_accel(accel: np.ndarray) -> Tuple[float, float]:
    """
    Extract roll and pitch angles from accelerometer data using gravity vector.
    
    Args:
        accel: 3D acceleration vector (m/s²) including gravity
        
    Returns:
        (roll, pitch) in radians
    """
    # Normalize accelerometer reading
    accel_norm = np.linalg.norm(accel)
    if accel_norm < 1e-6:
        return 0.0, 0.0
        
    a = accel / accel_norm
    
    # Roll: rotation around X-axis (atan2(ay, az))
    roll = np.arctan2(a[1], a[2])
    
    # Pitch: rotation around Y-axis (atan2(-ax, sqrt(ay² + az²)))
    pitch = np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2))
    
    return roll, pitch


@dataclass
class MotionAnalysis:
    """Results from analyzing motion data."""
    gyro_data: np.ndarray  # N x 3 array of gyro readings
    accel_angular_rates: np.ndarray  # N x 3 array of angular rates from accel
    correlations: np.ndarray  # 3x3 correlation matrix
    axis_mapping: Dict[int, int]  # Maps gyro axis to best matching accel-derived axis
    sign_corrections: Dict[int, float]  # Sign corrections needed (+1 or -1)
    confidence_scores: np.ndarray  # Confidence in each axis mapping


class CoordinateFrameDetector:
    """Automatic coordinate frame transformation detector."""
    
    def __init__(self, min_motion_threshold: float = 0.1, dt_estimate: float = 0.033):
        """
        Initialize detector.
        
        Args:
            min_motion_threshold: Minimum motion required to consider a sample (rad/s)
            dt_estimate: Estimated time step for numerical differentiation (seconds)
        """
        self.min_motion_threshold = min_motion_threshold
        self.dt_estimate = dt_estimate
        self.gyro_samples = []
        self.accel_samples = []
        self.timestamps = []
        
    def add_sample(self, gyro: np.ndarray, accel: np.ndarray, timestamp: float):
        """Add a new IMU sample to the analysis."""
        # Only keep samples with sufficient gyro motion
        gyro_magnitude = np.linalg.norm(gyro)
        
        if gyro_magnitude > self.min_motion_threshold:
            self.gyro_samples.append(gyro.copy())
            self.accel_samples.append(accel.copy())
            self.timestamps.append(timestamp)
    
    def analyze_angular_rate_correlations(self) -> MotionAnalysis:
        """
        Analyze the collected motion data by comparing gyro angular rates
        with angular rates derived from accelerometer tilt changes.
        """
        if len(self.gyro_samples) < 50:
            raise ValueError(f"Need at least 50 motion samples, got {len(self.gyro_samples)}")
        
        gyro_data = np.array(self.gyro_samples)
        accel_data = np.array(self.accel_samples)
        timestamps = np.array(self.timestamps)
        
        print(f"Analyzing {len(self.gyro_samples)} motion samples...")
        
        # Extract roll/pitch from accelerometer at each time step
        roll_pitch_data = np.zeros((len(accel_data), 2))
        for i, accel in enumerate(accel_data):
            roll, pitch = extract_roll_pitch_from_accel(accel)
            roll_pitch_data[i] = [roll, pitch]
        
        # Compute angular rates from accelerometer by differentiating roll/pitch
        accel_angular_rates = np.zeros((len(accel_data), 3))
        
        # Use numerical differentiation for roll/pitch rates
        for i in range(1, len(roll_pitch_data)):
            dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # Convert ms to seconds
            if dt > 0:
                roll_rate = (roll_pitch_data[i, 0] - roll_pitch_data[i-1, 0]) / dt
                pitch_rate = (roll_pitch_data[i, 1] - roll_pitch_data[i-1, 1]) / dt
                
                # Handle angle wrap-around for better differentiation
                if abs(roll_rate) > np.pi / dt:
                    if roll_rate > 0:
                        roll_rate -= 2*np.pi / dt
                    else:
                        roll_rate += 2*np.pi / dt
                        
                if abs(pitch_rate) > np.pi / dt:
                    if pitch_rate > 0:
                        pitch_rate -= 2*np.pi / dt
                    else:
                        pitch_rate += 2*np.pi / dt
                
                # Store roll rate (X rotation) and pitch rate (Y rotation)
                # Yaw rate (Z rotation) cannot be determined from accelerometer alone
                accel_angular_rates[i, 0] = roll_rate
                accel_angular_rates[i, 1] = pitch_rate
                accel_angular_rates[i, 2] = 0.0  # Yaw rate unknown
        
        # Remove first sample (no derivative available)
        gyro_data = gyro_data[1:]
        accel_angular_rates = accel_angular_rates[1:]
        
        print(f"Computed angular rates from accelerometer tilt changes")
        print(f"Gyro data range: X=[{np.min(gyro_data[:,0]):.3f}, {np.max(gyro_data[:,0]):.3f}], "
              f"Y=[{np.min(gyro_data[:,1]):.3f}, {np.max(gyro_data[:,1]):.3f}], "
              f"Z=[{np.min(gyro_data[:,2]):.3f}, {np.max(gyro_data[:,2]):.3f}] rad/s")
        print(f"Accel-derived rates: X=[{np.min(accel_angular_rates[:,0]):.3f}, {np.max(accel_angular_rates[:,0]):.3f}], "
              f"Y=[{np.min(accel_angular_rates[:,1]):.3f}, {np.max(accel_angular_rates[:,1]):.3f}] rad/s")
        
        # Compute correlations between gyro axes and accel-derived angular rates
        correlations = np.zeros((3, 3))
        
        for gyro_axis in range(3):
            for accel_axis in range(2):  # Only X,Y available from accel (roll, pitch)
                # Compute correlation coefficient
                corr = np.corrcoef(gyro_data[:, gyro_axis], accel_angular_rates[:, accel_axis])[0, 1]
                if not np.isnan(corr):
                    correlations[gyro_axis, accel_axis] = corr
        
        print("\nCorrelation matrix (gyro axes vs accel-derived angular rates):")
        print("       Roll_Rate  Pitch_Rate  (Yaw_N/A)")
        for i in range(3):
            axis_name = ['Gyro_X', 'Gyro_Y', 'Gyro_Z'][i]
            print(f"{axis_name}     {correlations[i, 0]:+6.3f}    {correlations[i, 1]:+6.3f}        N/A")
        
        # Find best axis mapping
        axis_mapping = {}
        sign_corrections = {}
        confidence_scores = np.zeros(3)
        used_accel_axes = set()
        
        # Use a greedy approach: find the strongest correlations first
        correlation_candidates = []
        for gyro_axis in range(3):
            for accel_axis in range(2):  # Only roll and pitch rates
                abs_corr = abs(correlations[gyro_axis, accel_axis])
                if abs_corr > 0.3:  # Threshold for angular rates
                    correlation_candidates.append((abs_corr, gyro_axis, accel_axis))
        
        # Sort by correlation strength
        correlation_candidates.sort(reverse=True)
        
        for abs_corr, gyro_axis, accel_axis in correlation_candidates:
            if gyro_axis not in axis_mapping and accel_axis not in used_accel_axes:
                axis_mapping[gyro_axis] = accel_axis
                sign_corrections[gyro_axis] = np.sign(correlations[gyro_axis, accel_axis])
                confidence_scores[gyro_axis] = abs_corr
                used_accel_axes.add(accel_axis)
                
                rate_type = "Roll_Rate" if accel_axis == 0 else "Pitch_Rate"
                sign_str = "+" if sign_corrections[gyro_axis] > 0 else "-"
                print(f"  Gyro_{['X','Y','Z'][gyro_axis]} -> {sign_str}{rate_type} (confidence: {abs_corr:.3f})")
        
        if len(axis_mapping) == 0:
            print("  Warning: No reliable axis mappings found")
        
        return MotionAnalysis(
            gyro_data=gyro_data,
            accel_angular_rates=accel_angular_rates,
            correlations=correlations,
            axis_mapping=axis_mapping,
            sign_corrections=sign_corrections,
            confidence_scores=confidence_scores
        )
    
    def generate_transform_matrix(self, analysis: MotionAnalysis, 
                                target: str = 'accel') -> Tuple[np.ndarray, Dict]:
        """
        Generate coordinate transformation matrix based on analysis.
        
        The analysis tells us which gyro axis measures which type of rotation:
        - Gyro_X correlates with Roll_Rate  -> Gyro_X measures rotation around some axis
        - Gyro_Z correlates with Pitch_Rate -> Gyro_Z measures rotation around some axis
        
        But we need to be careful: correlation with roll/pitch RATE doesn't directly
        tell us the coordinate frame transformation. We need to think about what
        coordinate system would produce these correlations.
        
        Args:
            analysis: Results from motion analysis
            target: 'accel' to transform accel to gyro frame, or 'gyro' for reverse
        
        Returns:
            (transform_matrix, debug_info)
        """
        debug_info = {
            'mappings': [],
            'confidence': 0.0,
            'warnings': [],
            'interpretation': {}
        }
        
        if len(analysis.axis_mapping) == 0:
            debug_info['warnings'].append("No axis mappings found")
            return np.eye(3), debug_info
        
        # Interpret the results
        print("\n=== Interpreting the correlations ===")
        for gyro_axis, accel_rate_axis in analysis.axis_mapping.items():
            sign = analysis.sign_corrections[gyro_axis]
            confidence = analysis.confidence_scores[gyro_axis]
            
            gyro_name = ['X', 'Y', 'Z'][gyro_axis]
            rate_name = ['Roll', 'Pitch'][accel_rate_axis]
            rotation_axis = ['X', 'Y'][accel_rate_axis]  # Roll=X rotation, Pitch=Y rotation
            
            print(f"Gyro_{gyro_name} correlates with {rate_name}_Rate (rotation around {rotation_axis}-axis)")
            print(f"  Sign: {'+' if sign > 0 else '-'}, Confidence: {confidence:.3f}")
            
            debug_info['interpretation'][f'gyro_{gyro_name.lower()}'] = {
                'measures_rotation_around': rotation_axis,
                'sign': sign,
                'confidence': confidence
            }
        
        # For now, let's be honest about what we can determine
        print(f"\nIMPORTANT: The correlations tell us which gyro axis measures which type of rotation,")
        print(f"but determining the exact coordinate frame transformation requires more analysis.")
        print(f"The current approach gives us the ORIENTATION of the IMU relative to a standard frame.")
        
        # Generate a simple diagnostic matrix that shows the detected relationships
        transform = np.eye(3)
        
        # Add diagnostic information
        detected_relationships = []
        for gyro_axis, accel_rate_axis in analysis.axis_mapping.items():
            sign = analysis.sign_corrections[gyro_axis]
            gyro_name = ['X', 'Y', 'Z'][gyro_axis]
            rate_name = ['Roll', 'Pitch'][accel_rate_axis]
            detected_relationships.append(f"Gyro_{gyro_name} -> {'+' if sign > 0 else '-'}{rate_name}")
        
        debug_info['detected_relationships'] = detected_relationships
        debug_info['warnings'].append("Identity matrix returned - transform logic needs refinement")
        debug_info['warnings'].append("Correlations show axis relationships but not direct frame transform")
        
        # Calculate overall confidence
        valid_confidences = [analysis.confidence_scores[i] for i in analysis.axis_mapping.keys()]
        debug_info['confidence'] = np.mean(valid_confidences) if valid_confidences else 0.0
        
        return transform, debug_info


def collect_motion_data(subscriber, processor, duration_seconds: float = 30) -> CoordinateFrameDetector:
    """
    Collect motion data for coordinate frame detection.
    
    Args:
        subscriber: ZenohD435iSubscriber instance
        processor: IMUProcessor instance
        duration_seconds: How long to collect data
    
    Returns:
        CoordinateFrameDetector with collected data
    """
    print(f"=== Motion Data Collection ({duration_seconds}s) ===")
    print("Please move the device to generate angular motion:")
    print("1. Slowly rotate around different axes")
    print("2. Tilt forward/backward and left/right") 
    print("3. Focus on ROLL and PITCH motions (avoid pure yaw)")
    print("4. Make smooth, deliberate movements")
    print("\nCollecting data...")
    
    detector = CoordinateFrameDetector()
    start_time = time.time()
    last_frame_id = -1
    sample_count = 0
    
    # Wait for initial data
    while True:
        frame_data = subscriber.get_latest_frames()
        if frame_data.frame_count > 0 and frame_data.motion is not None:
            break
        time.sleep(0.1)
    
    print("Data collection started - move the device now!")
    
    while time.time() - start_time < duration_seconds:
        frame_data = subscriber.get_latest_frames()
        
        if frame_data.frame_count == 0 or frame_data.frame_count == last_frame_id:
            time.sleep(0.001)
            continue
            
        if frame_data.motion is None:
            continue
            
        last_frame_id = frame_data.frame_count
        
        # Process IMU data
        raw_gyro = np.array(frame_data.motion.gyro)
        raw_accel = np.array(frame_data.motion.accel)
        timestamp = frame_data.motion.timestamp
        
        cal_gyro, cal_accel = processor.process_sample(raw_gyro, raw_accel)
        
        # Add to detector
        detector.add_sample(cal_gyro, cal_accel, timestamp)
        sample_count += 1
        
        # Progress indicator
        elapsed = time.time() - start_time
        if sample_count % 100 == 0:
            remaining = duration_seconds - elapsed
            motion_samples = len(detector.gyro_samples)
            print(f"  Progress: {elapsed:.1f}/{duration_seconds}s, "
                  f"Total samples: {sample_count}, Motion samples: {motion_samples}, "
                  f"Remaining: {remaining:.1f}s")
    
    print(f"\nData collection complete!")
    print(f"  Total samples processed: {sample_count}")
    print(f"  Motion samples collected: {len(detector.gyro_samples)}")
    
    return detector


def test_detected_transforms(subscriber, processor, accel_transform, gyro_transform, 
                           test_duration: float = 10) -> Dict:
    """
    Test the detected coordinate transforms by running orientation estimation.
    
    Returns:
        Dictionary with test results and quality metrics
    """
    print(f"\n=== Testing Detected Transforms ({test_duration}s) ===")
    print("Move the device and observe if gyro and accel estimates align better...")
    
    # Create estimator with detected transforms
    estimator = OrientationEstimator(
        accel_frame_transform=accel_transform,
        gyro_frame_transform=gyro_transform,
        k_p=2.0, k_i=0.1
    )
    
    # Create reference estimator without transforms
    reference_estimator = OrientationEstimator(k_p=2.0, k_i=0.1)
    
    start_time = time.time()
    last_frame_id = -1
    last_timestamp = None
    
    # Metrics
    angle_differences = []
    reference_differences = []
    accel_usage_rates = []
    
    while time.time() - start_time < test_duration:
        frame_data = subscriber.get_latest_frames()
        
        if frame_data.frame_count == 0 or frame_data.frame_count == last_frame_id:
            time.sleep(0.01)
            continue
            
        if frame_data.motion is None:
            continue
            
        last_frame_id = frame_data.frame_count
        
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
        
        # Update both estimators
        estimator.update(cal_gyro, cal_accel, dt)
        reference_estimator.update(cal_gyro, cal_accel, dt)
        
        # Get orientations
        gyro_only, accel_only, fused = estimator.get_all_orientations()
        ref_gyro_only, ref_accel_only, ref_fused = reference_estimator.get_all_orientations()
        
        # Measure alignment quality
        gyro_euler = np.degrees(gyro_only.to_euler_angles())
        accel_euler = np.degrees(accel_only.to_euler_angles())
        ref_gyro_euler = np.degrees(ref_gyro_only.to_euler_angles())
        ref_accel_euler = np.degrees(ref_accel_only.to_euler_angles())
        
        # Calculate roll/pitch differences (ignore yaw as accel can't measure it)
        rp_diff = np.sqrt(np.mean((gyro_euler[:2] - accel_euler[:2])**2))
        ref_rp_diff = np.sqrt(np.mean((ref_gyro_euler[:2] - ref_accel_euler[:2])**2))
        
        angle_differences.append(rp_diff)
        reference_differences.append(ref_rp_diff)
        
        # Track accelerometer usage
        debug_stats = estimator.get_debug_stats()
        accel_usage_rates.append(debug_stats['accel_usage_rate'])
    
    # Calculate results
    avg_difference = np.mean(angle_differences)
    ref_avg_difference = np.mean(reference_differences)
    improvement = ref_avg_difference - avg_difference
    improvement_pct = (improvement / ref_avg_difference) * 100 if ref_avg_difference > 0 else 0
    avg_accel_usage = np.mean(accel_usage_rates)
    
    results = {
        'avg_rp_difference': avg_difference,
        'reference_rp_difference': ref_avg_difference,
        'improvement_degrees': improvement,
        'improvement_percentage': improvement_pct,
        'avg_accel_usage': avg_accel_usage,
        'samples_tested': len(angle_differences)
    }
    
    print(f"\nTest Results:")
    print(f"  With transforms - Avg Roll/Pitch difference: {avg_difference:.2f}°")
    print(f"  Without transforms - Avg Roll/Pitch difference: {ref_avg_difference:.2f}°")
    print(f"  Improvement: {improvement:.2f}° ({improvement_pct:+.1f}%)")
    print(f"  Accelerometer usage rate: {avg_accel_usage:.1f}%")
    
    return results


def main():
    """Main coordinate frame detection program."""
    print("=== Automatic IMU Coordinate Frame Detection ===")
    print("This program automatically detects coordinate frame misalignments")
    print("by comparing gyroscope angular rates with angular rates derived")
    print("from accelerometer tilt changes.\n")
    
    # Load calibration
    calibration = load_calibration()
    if calibration is None:
        print("Warning: No calibration found. Results may be less accurate.")
    else:
        print("✓ Using calibrated IMU data")
    
    # Set up subscriber and processor
    subscriber = zd435i.ZenohD435iSubscriber()
    processor = IMUProcessor(calibration)
    
    print("\nConnecting to Zenoh...")
    try:
        subscriber.connect()
        subscriber.start_subscribing()
        print("✓ Connected and subscribed")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    try:
        # Step 1: Collect motion data
        detector = collect_motion_data(subscriber, processor, duration_seconds=30)
        
        if len(detector.gyro_samples) < 50:
            print(f"✗ Insufficient motion data collected ({len(detector.gyro_samples)} samples)")
            print("Please move the device more during data collection!")
            return
        
        # Step 2: Analyze angular rate correlations
        print("\n=== Analyzing Angular Rate Correlations ===")
        analysis = detector.analyze_angular_rate_correlations()
        
        print(f"\nDetected axis mappings:")
        for gyro_axis, accel_rate_axis in analysis.axis_mapping.items():
            sign = analysis.sign_corrections[gyro_axis]
            confidence = analysis.confidence_scores[gyro_axis]
            sign_str = "+" if sign > 0 else "-"
            rate_type = "Roll_Rate" if accel_rate_axis == 0 else "Pitch_Rate"
            axis_names = ['X', 'Y', 'Z']
            print(f"  Gyro_{axis_names[gyro_axis]} -> {sign_str}{rate_type} "
                  f"(confidence: {confidence:.3f})")
        
        # Step 3: Generate transformation matrices
        print("\n=== Generating Transformation Matrices ===")
        accel_transform, accel_debug = detector.generate_transform_matrix(analysis, 'accel')
        
        print("Accelerometer transformation matrix (to align with gyro frame):")
        print(accel_transform)
        print(f"Confidence: {accel_debug['confidence']:.3f}")
        
        if accel_debug['warnings']:
            print("Warnings:")
            for warning in accel_debug['warnings']:
                print(f"  ⚠ {warning}")
        
        # Step 4: Test the transformations
        test_results = test_detected_transforms(subscriber, processor, 
                                              accel_transform, None, test_duration=15)
        
        # Step 5: Provide recommendations
        print("\n=== Recommendations ===")
        
        if test_results['improvement_percentage'] > 20:
            print("✓ Excellent improvement! The detected transforms work well.")
            print("  Recommended: Use the accelerometer transform in your IMU processing.")
        elif test_results['improvement_percentage'] > 10:
            print("✓ Good improvement. The detected transforms help.")
            print("  Recommended: Use the transforms, but consider fine-tuning.")
        elif test_results['improvement_percentage'] > 0:
            print("⚠ Marginal improvement. Transforms help slightly.")
            print("  Consider: Manual verification or collecting more motion data.")
        else:
            print("✗ No significant improvement detected.")
            print("  Possible causes: Insufficient motion, noise, or already aligned frames.")
        
        print(f"\nTo use these transforms in your code:")
        print(f"```python")
        print(f"import numpy as np")
        print(f"accel_transform = np.array({accel_transform.tolist()})")
        print(f"estimator = OrientationEstimator(accel_frame_transform=accel_transform)")
        print(f"```")
        
        # Save results
        save_path = "detected_transforms.json"
        import json
        results_dict = {
            'accel_transform': accel_transform.tolist(),
            'analysis': {
                'axis_mapping': {str(k): int(v) for k, v in analysis.axis_mapping.items()},
                'sign_corrections': {str(k): float(v) for k, v in analysis.sign_corrections.items()},
                'confidence_scores': analysis.confidence_scores.tolist(),
                'correlations': analysis.correlations.tolist()
            },
            'test_results': test_results,
            'timestamp': time.time(),
            'method': 'angular_rate_correlation'
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {save_path}")
        
    except KeyboardInterrupt:
        print("\nDetection interrupted by user.")
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        subscriber.stop()
        print("Done!")


if __name__ == "__main__":
    main() 