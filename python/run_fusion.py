#!/usr/bin/env python3
"""
Convenience script to run the fusion system in different modes.
"""
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run sensor fusion system")
    parser.add_argument('mode', choices=['live', 'test', 'calibrate', 'gyro-only', 'vo-only'],
                       help='Mode to run the system in')
    parser.add_argument('--record', action='store_true',
                       help='Record sensor data to file (for live mode)')
    parser.add_argument('--playback', type=str, metavar='FILE',
                       help='Playback recorded data from file')
    
    args = parser.parse_args()
    
    # Change to python directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if args.mode == 'live':
        # Run full fusion system with live data
        cmd = ['python3', 'fusion_pose_estimate.py']
        if args.record:
            cmd.extend(['--record'])
        elif args.playback:
            cmd.extend(['--playback', args.playback])
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.mode == 'test':
        # Run synthetic test
        print("Running synthetic data test...")
        subprocess.run(['python3', 'fusion_test_utils.py'])
        
    elif args.mode == 'calibrate':
        # Run IMU calibration
        print("Running IMU calibration...")
        subprocess.run(['python3', 'calibrate_offsets.py'])
        
    elif args.mode == 'gyro-only':
        # Run gyro-only estimation
        print("Running gyroscope-only angle estimation...")
        subprocess.run(['python3', 'gyro_angle_estimate.py'])
        
    elif args.mode == 'vo-only':
        # Run visual odometry only
        print("Running visual odometry only...")
        cmd = ['python3', 'alignment_and_matching.py']
        if args.record:
            cmd.extend(['--record'])
        elif args.playback:
            cmd.extend(['--playback', args.playback])
        subprocess.run(cmd)

if __name__ == "__main__":
    main() 