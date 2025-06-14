#!/usr/bin/env python3
"""
Example usage of the Zenoh D435i Subscriber Python package.
Simple polling-based API - just call get_latest_frames() to get the latest data!
"""

import zenoh_d435i_subscriber as zd435i
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    print("🚀 Starting Zenoh D435i Subscriber Example")
    print("📦 Available classes:", [name for name in dir(zd435i) if not name.startswith('_')])
    
    # Create subscriber
    subscriber = zd435i.ZenohD435iSubscriber()
    print(f"✅ Created subscriber: {subscriber}")
    
    # Set up camera intrinsics (optional)
    depth_intrinsics = zd435i.PyIntrinsics(
        width=640, height=480,
        fx=387.31454, fy=387.31454,
        ppx=322.1206, ppy=236.50139
    )
    
    color_intrinsics = zd435i.PyIntrinsics(
        width=640, height=480,
        fx=607.2676, fy=607.149,
        ppx=316.65408, ppy=244.13338
    )
    
    subscriber.set_depth_intrinsics(depth_intrinsics)
    subscriber.set_color_intrinsics(color_intrinsics)
    print(f"📷 Set intrinsics - Depth: {depth_intrinsics}")
    print(f"📷 Set intrinsics - Color: {color_intrinsics}")
    
    # Flag to track if we've plotted the first depth image
    first_depth_plotted = False
    
    try:
        # Connect to Zenoh
        print("🔌 Connecting to Zenoh...")
        subscriber.connect()
        print("✅ Connected!")
        
        # Start subscribing
        print("📡 Starting subscription...")
        subscriber.start_subscribing()
        print("✅ Subscription started!")
        
        print("\n" + "="*60)
        print("📊 LIVE DATA STREAM")
        print("="*60)
        print("💡 The subscriber is now running in the background.")
        print("💡 We'll poll for the latest data every second.")
        print("💡 The first depth image will be plotted and saved.")
        print("💡 Press Ctrl+C to stop.\n")
        
        last_frame_count = 0
        
        while subscriber.is_running():
            # Get the latest frames (this is non-blocking!)
            frame_data = subscriber.get_latest_frames()
            
            # Check if we got new data
            if frame_data.frame_count > last_frame_count:
                print(f"🎯 NEW DATA! {frame_data}")
                
                # Process RGB data if available
                if frame_data.rgb:
                    rgb_data = frame_data.rgb.get_data()
                    print(f"  📸 RGB: {len(rgb_data)} bytes ({frame_data.rgb.width}x{frame_data.rgb.height})")
                
                # Process Depth data if available
                if frame_data.depth and not first_depth_plotted:
                    depth_array = frame_data.depth.get_data()
                    depth_2d = frame_data.depth.get_data_2d()
                    depth_min, depth_max = np.min(depth_array), np.max(depth_array)
                    print(f"  🎯 Depth: {depth_2d.shape} array, range: {depth_min:.3f}-{depth_max:.3f}m")
                    
                    # Plot the first depth image
                    print("  📊 Plotting first depth image...")
                    plt.figure(figsize=(12, 8))
                    
                    # Create subplot for depth visualization
                    plt.subplot(1, 2, 1)
                    plt.imshow(depth_2d, cmap='viridis', vmin=0, vmax=np.percentile(depth_2d[depth_2d > 0], 95))
                    plt.colorbar(label='Depth (m)')
                    plt.title(f'First Depth Image\n{depth_2d.shape[1]}x{depth_2d.shape[0]} pixels')
                    plt.xlabel('X (pixels)')
                    plt.ylabel('Y (pixels)')
                    
                    # Create histogram of depth values
                    plt.subplot(1, 2, 2)
                    valid_depths = depth_2d[depth_2d > 0]  # Only non-zero depths
                    if len(valid_depths) > 0:
                        plt.hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
                        plt.xlabel('Depth (m)')
                        plt.ylabel('Pixel Count')
                        plt.title(f'Depth Distribution\nValid pixels: {len(valid_depths):,}')
                        plt.axvline(np.mean(valid_depths), color='red', linestyle='--', 
                                   label=f'Mean: {np.mean(valid_depths):.3f}m')
                        plt.axvline(np.median(valid_depths), color='orange', linestyle='--', 
                                   label=f'Median: {np.median(valid_depths):.3f}m')
                        plt.legend()
                    
                    plt.tight_layout()
                    
                    # Save the plot
                    filename = f'first_depth_image_{int(time.time())}.png'
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"  💾 Saved depth image as: {filename}")
                    
                    # Show the plot
                    # plt.show()
                    
                    first_depth_plotted = True
                    print("  ✅ First depth image plotted and saved!")
                
                elif frame_data.depth:
                    # Just print stats for subsequent depth frames
                    depth_array = frame_data.depth.get_data()
                    depth_2d = frame_data.depth.get_data_2d()
                    depth_min, depth_max = np.min(depth_array), np.max(depth_array)
                    print(f"  🎯 Depth: {depth_2d.shape} array, range: {depth_min:.3f}-{depth_max:.3f}m")
                
                # Process Motion data if available
                if frame_data.motion:
                    gyro = np.array(frame_data.motion.gyro)
                    accel = np.array(frame_data.motion.accel)
                    print(f"  🌀 Motion: gyro={gyro}, accel={accel}")
                
                last_frame_count = frame_data.frame_count
                print()
            else:
                # No new data, just show we're alive
                if frame_data.frame_count == 0:
                    print("⏳ Waiting for first frame...")
                else:
                    print(f"📊 {subscriber.get_stats()}")
            
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping subscriber...")
        subscriber.stop()
        print("✅ Stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("🏁 Example finished.")

if __name__ == "__main__":
    main() 