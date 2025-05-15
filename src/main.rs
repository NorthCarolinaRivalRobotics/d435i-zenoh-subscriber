use rerun_utils::log_depth;
use types::{ColorFrameSerializable, DepthFrameSerializable};
use std::time::{Duration, Instant};
use snap::raw::Decoder;
mod types;
mod rerun_utils;

#[tokio::main]
async fn main() {
    let session = zenoh::open(zenoh::Config::default()).await.unwrap();
    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();
    let depthSubscriber = session.declare_subscriber("camera/depth").await.unwrap();
    let colorSubscriber = session.declare_subscriber("camera/rgb").await.unwrap();

    let mut last_print_time = Instant::now();
    let mut frame_count = 0u32;
    const FPS_PRINT_INTERVAL: Duration = Duration::from_secs(1);

    loop {
        let loop_start_time = Instant::now();
        tokio::select! {
            sample = depthSubscriber.recv_async() => {
                if let Ok(sample) = sample {
                    let recv_depth_elapsed = loop_start_time.elapsed();
                    println!("Depth wait time (approx): {:?}", recv_depth_elapsed);

                    let decode_start = Instant::now();
                    let depth_frame = DepthFrameSerializable::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                    let decode_duration = decode_start.elapsed();
                    println!("Depth decode time: {:?}", decode_duration);

                    let log_depth_start = Instant::now();
                    log_depth(&rec, &depth_frame).unwrap();
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
                } else {
                    break;
                }
            }
            sample = colorSubscriber.recv_async() => {
                if let Ok(sample) = sample {
                    let recv_color_elapsed = loop_start_time.elapsed();
                    println!("Color wait time (approx): {:?}", recv_color_elapsed);

                    let decode_start = Instant::now();
                    let color_frame = ColorFrameSerializable::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                    let decode_duration = decode_start.elapsed();
                    println!("Color decode time: {:?}", decode_duration);
                    println!("Received (color): {:?}", color_frame.data.len());
                } else {
                    break;
                }
            }
        }
    }
}