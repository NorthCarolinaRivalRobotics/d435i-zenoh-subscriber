use nalgebra::{Matrix3, Vector3};
use rerun_utils::{log_aligned_depth, log_depth, log_rgb_jpeg};
use types::{ColorFrameSerializable, DepthFrameSerializable, Extrinsics, Frame, Intrinsics, MotionFrameData};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::unbounded_channel;
use clap::Parser;
use std::sync::Arc;
mod types;
mod rerun_utils;
mod reproject;
mod sync;
mod cli;
mod odometry;
mod logio;

#[tokio::main]
async fn main() {
    let args = cli::Args::parse();
    let mode = args.mode();

    // channel graph:
    // [zenoh OR playback] -> sync -> odometry
    let (raw_tx, raw_rx) = unbounded_channel::<Frame>();
    let (odom_tx, odom_rx) = unbounded_channel::<odometry::StampedTriple>();

    // 1. spawn synchroniser
    tokio::spawn(sync::run(raw_rx, odom_tx));

    // 2. spawn odometry consumer
    tokio::spawn(odometry::run(odom_rx));

    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();

    // create if we're in a live mode, otherwise None
    let live_session = if matches!(mode, cli::RunMode::Playback(_)) {
        None
    } else {
        Some(zenoh::open(zenoh::Config::default()).await.unwrap())
    };

    // 3. live or playback source
    match mode.clone() {
        cli::RunMode::Playback(path) => {
            tokio::spawn(async move {
                logio::reader::playback(&path, raw_tx).await.unwrap();
            });
        }
        _ => {
            // ensure session exists before using it
            let session = live_session.as_ref().unwrap();
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

            // Optional log path that will be used to create loggers for each task
            let log_path = match &mode {
                cli::RunMode::LiveWithLog(path) => Some(path.clone()),
                _ => None,
            };

            // depth task
            let tx = raw_tx.clone();
            let rec_depth = rec.clone();
            let depth_log_path = log_path.clone();
            let depth_task = tokio::spawn(async move {
                // Create logger if path was provided
                let mut depth_logger = match depth_log_path {
                    Some(path) => Some(logio::writer::LogWriter::new(&path).unwrap()),
                    None => None,
                };
                
                let mut last_print_time = Instant::now();
                let mut frame_count = 0u32;
                const FPS_PRINT_INTERVAL: Duration = Duration::from_secs(1);

                loop {
                    match depth_subscriber.recv_async().await {
                        Ok(sample) => {

                            let depth_frame = DepthFrameSerializable::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                            let reprojected_depth = reproject::align_depth_to_color(&depth_frame, &d_intr, &c_intr, &extr);

                            // Send to channel
                            let frame = Frame::Depth(depth_frame.clone());
                            let _ = tx.send(frame.clone());
                            if let Some(writer) = depth_logger.as_mut() { writer.write(&frame).unwrap(); }

                            // Log to rerun for visualization
                            log_depth(&rec_depth, &depth_frame).unwrap();
                            log_aligned_depth(&rec_depth, &reprojected_depth.data, reprojected_depth.width, reprojected_depth.height).unwrap();
                            
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

            // color task
            let tx = raw_tx.clone();
            let rec_color = rec.clone();
            let color_log_path = log_path.clone();
            let color_task = tokio::spawn(async move {
                // Create logger if path was provided
                let mut color_logger = match color_log_path {
                    Some(path) => Some(logio::writer::LogWriter::new(&path).unwrap()),
                    None => None,
                };
                
                loop {
                    match color_subscriber.recv_async().await {
                        Ok(sample) => {
                            let (color_frame, timestamp) = ColorFrameSerializable::decodeAndDecompress(sample.payload().to_bytes().to_vec());
                            
                            // Send to channel
                            let frame = Frame::Color((color_frame.clone(), timestamp));
                            let _ = tx.send(frame.clone());
                            if let Some(writer) = color_logger.as_mut() { writer.write(&frame).unwrap(); }
                            
                            // Log to rerun for visualization
                            log_rgb_jpeg(&rec_color, &color_frame).unwrap();
                        }
                        Err(_) => {
                            println!("Color stream closed");
                            break;
                        }
                    }
                }
            });

            // motion task
            let tx = raw_tx.clone();
            let motion_log_path = log_path.clone();
            let motion_task = tokio::spawn(async move {
                // Create logger if path was provided
                let mut motion_logger = match motion_log_path {
                    Some(path) => Some(logio::writer::LogWriter::new(&path).unwrap()),
                    None => None,
                };
                
                loop {
                    match motion_subscriber.recv_async().await {
                        Ok(sample) => {
                            let motion_frame = MotionFrameData::decodeAndDecompress(sample.payload().to_bytes().to_vec());                            
                            // Send to channel
                            let frame = Frame::Motion(motion_frame.clone());
                            let _ = tx.send(frame.clone());
                            if let Some(writer) = motion_logger.as_mut() { writer.write(&frame).unwrap(); }
                        }
                        Err(_) => {
                            println!("Motion stream closed");
                            break;
                        }
                    }
                }
            });
        }
    }

    // keep main alive forever
    tokio::signal::ctrl_c().await.unwrap();
}

