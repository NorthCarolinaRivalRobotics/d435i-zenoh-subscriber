use rerun_utils::{log_depth, log_rgb_jpeg};
use types::{ColorFrameSerializable, DepthFrameSerializable};
use std::time::{Duration, Instant};
use snap::raw::Decoder;
mod types;
mod rerun_utils;

#[tokio::main]
async fn main() {
    let session = zenoh::open(zenoh::Config::default()).await.unwrap();
    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();
    let depth_subscriber = session.declare_subscriber("camera/depth").await.unwrap();
    let color_subscriber = session.declare_subscriber("camera/rgb").await.unwrap();

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
                    let decode_duration = decode_start.elapsed();
                    println!("Depth decode time: {:?}", decode_duration);

                    let log_depth_start = Instant::now();
                    log_depth(&rec_depth, &depth_frame).unwrap();
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

    // Wait for both tasks to complete
    // If they are infinite loops, this will run indefinitely unless one of them errors out or breaks.
    let _ = tokio::try_join!(depth_task, color_task);
}

