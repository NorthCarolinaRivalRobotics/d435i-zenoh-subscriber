use nalgebra::{Matrix3, Vector3};
use rerun_utils::{log_aligned_depth, log_depth, log_rgb_jpeg};
use types::{ColorFrameSerializable, DepthFrameSerializable, Extrinsics, Intrinsics, MotionFrameData};
use std::time::{Duration, Instant};
use snap::raw::Decoder;
mod types;
mod rerun_utils;
mod reproject;


#[tokio::main]
async fn main() {
    let session = zenoh::open(zenoh::Config::default()).await.unwrap();
    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();
    let depth_subscriber = session.declare_subscriber("camera/depth").await.unwrap();
    let color_subscriber = session.declare_subscriber("camera/rgb").await.unwrap();
    let motion_subscriber = session.declare_subscriber("camera/motion").await.unwrap();

    let d_intr = Intrinsics { width: 640, height: 480,
        fx: 387.31454, fy: 387.31454,
        ppx: 322.1206, ppy: 236.50139 };

    let c_intr = Intrinsics { width: 640, height: 480,
            fx: 607.2676, fy: 607.149,
            ppx: 316.65408, ppy: 244.13338 };


    let extr = Extrinsics {
        rotation: Matrix3::from_row_slice(&[
            0.9999627, -0.008320532,  0.0023323754,
            0.008310333,  0.999956,   0.0043491516,
        -0.00236846,  -0.0043296064, 0.99998784]),
        translation: Vector3::new(0.014476319, 0.0001452052, 0.00031550066),
    };


    let rec_depth = rec.clone();
    let depth_task = tokio::spawn(async move {
        let mut last_print_time = Instant::now();
        let mut frame_count = 0u32;
        const FPS_PRINT_INTERVAL: Duration = Duration::from_secs(1);

        loop {
            let loop_start_time = Instant::now();
            match depth_subscriber.recv_async().await {
                Ok(sample) => {
                    let recv_depth_elapsed = loop_start_time.elapsed();
                    println!("Depth wait time (approx): {:?}", recv_depth_elapsed);

                    let decode_start = Instant::now();
                    let depth_frame = DepthFrameSerializable::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                    let reprojected_depth = reproject::align_depth_to_color(&depth_frame, &d_intr, &c_intr, &extr);
                    let decode_duration = decode_start.elapsed();
                    println!("Depth decode time: {:?}", decode_duration);

                    let log_depth_start = Instant::now();
                    log_depth(&rec_depth, &depth_frame).unwrap();
                    log_aligned_depth(&rec_depth, &reprojected_depth.data, reprojected_depth.width, reprojected_depth.height).unwrap();
                    let log_depth_duration = log_depth_start.elapsed();
                    println!("log_depth time: {:?}", log_depth_duration);
                    
                    frame_count += 1;
                    if last_print_time.elapsed() >= FPS_PRINT_INTERVAL {
                        let elapsed_secs = last_print_time.elapsed().as_secs_f32();
                        let fps = frame_count as f32 / elapsed_secs;
                        println!("Depth FPS: {:.2}", fps);
                        frame_count = 0;
                        last_print_time = Instant::now();
                    }
                }
                Err(_) => {
                    println!("Depth stream closed");
                    break;
                }
            }
        }
    });

    let rec_color = rec.clone();
    let color_task = tokio::spawn(async move {
        loop {
            let loop_start_time = Instant::now();
            match color_subscriber.recv_async().await {
                Ok(sample) => {
                    let recv_color_elapsed = loop_start_time.elapsed();
                    println!("Color wait time (approx): {:?}", recv_color_elapsed);

                    let decode_start = Instant::now();
                    let (color_frame, timestamp) = ColorFrameSerializable::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                    let decode_duration = decode_start.elapsed();
                    println!("Color decode time: {:?}", decode_duration);
                    println!("Received (color): {:?}", color_frame.len());
                    println!("Timestamp: {:?}", timestamp);
                    log_rgb_jpeg(&rec_color, &color_frame).unwrap();
                }
                Err(_) => {
                    println!("Color stream closed");
                    break;
                }
            }
        }
    });

    let rec_motion = rec.clone();
    let motion_task = tokio::spawn(async move {
        loop {
            let loop_start_time = Instant::now();
            match motion_subscriber.recv_async().await {
                Ok(sample) => {
                    let motion_frame = MotionFrameData::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                    println!("Motion: {:?}", motion_frame);
                }
                Err(_) => {
                    println!("Motion stream closed");
                    break;
                }
            }
        }
    });
    


    // Wait for both tasks to complete
    // If they are infinite loops, this will run indefinitely unless one of them errors out or breaks.
    let _ = tokio::try_join!(depth_task, color_task, motion_task);
}

