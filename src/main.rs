use nalgebra::{Matrix3, Vector3};
use rerun_utils::{log_aligned_depth, log_depth, log_rgb_jpeg};
use types::{ColorFrameSerializable, DepthFrameSerializable, Extrinsics, Frame, Intrinsics, MotionFrameData, CombinedFrameWire, DepthFrameRealUnits};
use types::decode_u16_to_meters;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::unbounded_channel;
use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;
mod types;
mod rerun_utils;
mod reproject;
mod sync;
mod cli;
mod odometry;
mod logio;
mod rgbd_odom;

use std::{fs::File, io::{BufReader, Read}};
use byteorder::{LittleEndian, ReadBytesExt};
use zstd;
use bincode;
use turbojpeg;

fn read_sample_data() -> std::io::Result<()> {
    let mut f = BufReader::new(File::open("kitchen-loop.bin")?);
    let mut counts = [0usize; 3];
    while let Ok(kind) = f.read_u8() {
        let _ts  = f.read_f64::<LittleEndian>()?;
        let len  = f.read_u32::<LittleEndian>()? as usize;
        counts[kind as usize] += 1;
        // Skip the payload
        let mut buffer = vec![0u8; len];
        f.read_exact(&mut buffer)?;
    }
    println!("depth {:>5}, color {:>5}, motion {:>5}", counts[0], counts[1], counts[2]);
    Ok(())
}

/// Unpacks a combined frame into RGB and depth data
/// Returns (rgb_data, depth_data, width, height, timestamp) if successful
/// If either rgb or depth unpacking fails, returns None
fn unpack_combined_frame(frame: &CombinedFrameWire) -> Option<(Vec<u8>, Vec<u16>, u16, u16, f64)> {
    // 1. RGB decompression
    let rgb_raw = match turbojpeg::decompress_image::<turbojpeg::image::Rgb<u8>>(&frame.rgb_jpeg) {
        Ok(img) => img.into_raw(),
        Err(e) => {
            println!("RGB decompression error: {:?}", e);
            return None;
        }
    };
    
    // 2. Depth decompression and deserialization
    let depth_raw = match zstd::decode_all(&frame.depth_zstd[..]) {
        Ok(decompressed) => {
            match bincode::decode_from_slice::<DepthFrameSerializable, _>(&decompressed, bincode::config::standard()) {
                Ok((d, _)) => {
                    if d.data.is_empty() {
                        println!("Warning: Depth data contains 0 points");
                        return None;
                    }
                    d.data
                },
                Err(e) => {
                    println!("Depth deserialization error: {:?}", e);
                    return None;
                }
            }
        },
        Err(e) => {
            println!("Depth decompression error: {:?}", e);
            return None;
        }
    };
    
    Some((rgb_raw, depth_raw, frame.width, frame.height, frame.timestamp))
}

#[tokio::main]
async fn main() {
    let args = cli::Args::parse();
    let mode = args.mode();

    // channel graph:
    // [zenoh OR playback] -> sync -> odometry
    let (raw_tx, raw_rx) = unbounded_channel::<Frame>();
    let (odom_tx, odom_rx) = unbounded_channel::<odometry::StampedTriple>();

    // 1. spawn synchroniser
    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();
    tokio::spawn(sync::run(raw_rx, odom_tx));

    // 2. spawn odometry consumer
    tokio::spawn(odometry::run(odom_rx, rec.clone()));


    // create if we're in a live mode, otherwise None
    let live_session = if matches!(mode, cli::RunMode::Playback(_)) {
        None
    } else {
        Some(zenoh::open(zenoh::Config::default()).await.unwrap())
    };

    // Optional log path that will be used to create a shared logger
    let shared_logger = match &mode {
        cli::RunMode::LiveWithLog(path) => {
            Some(Arc::new(Mutex::new(logio::writer::LogWriter::new(path).unwrap())))
        },
        _ => None,
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
            let combined_subscriber = session.declare_subscriber("camera/combined").await.unwrap();
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

            // combined depth+color task
            let tx = raw_tx.clone();
            let rec_clone = rec.clone();
            let combined_logger = shared_logger.clone();
            let combined_task = tokio::spawn(async move {
                let mut last_print_time = Instant::now();
                let mut frame_count = 0u32;
                const FPS_PRINT_INTERVAL: Duration = Duration::from_secs(1);

                loop {
                    match combined_subscriber.recv_async().await {
                        Ok(sample) => {
                            let payload = sample.payload().to_bytes().to_vec();
                            
                            // 1️⃣ Write the raw packet immediately if logger exists
                            if let Some(logger) = &combined_logger {
                                // Use current time if sample timestamp is not available
                                let now = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs_f64();
                                
                                // Acquire mutex lock before writing
                                let mut log_writer = logger.lock().await;
                                log_writer.write_combined(now, &payload).unwrap();
                            }
                            
                            // 2️⃣ Normal unpacking & processing
                            let combined_frame = CombinedFrameWire::decode(&payload);
                            
                            // Unpack the combined frame
                            if let Some((rgb_raw, depth_raw, width, height, timestamp)) = unpack_combined_frame(&combined_frame) {
                                // Create a depth frame in real units
                                let depth_real_units = DepthFrameRealUnits {
                                    width: width as usize,
                                    height: height as usize,
                                    timestamp,
                                    data: depth_raw.iter().map(|&code| decode_u16_to_meters(code)).collect(),
                                };
                                
                                // Calculate reprojected depth
                                let reprojected_depth = reproject::align_depth_to_color(&depth_real_units, &d_intr, &c_intr, &extr);

                                // Send frames to channels
                                let depth_frame = Frame::Depth(reprojected_depth.clone());
                                let color_frame = Frame::Color((combined_frame.rgb_jpeg.clone(), timestamp));
                                
                                // Send to processing pipeline
                                let _ = tx.send(depth_frame.clone());
                                let _ = tx.send(color_frame.clone());
                                
                                // Log to rerun for visualization
                                log_depth(&rec_clone, &depth_real_units).unwrap();
                                log_aligned_depth(&rec_clone, &reprojected_depth.data, reprojected_depth.width, reprojected_depth.height).unwrap();
                                log_rgb_jpeg(&rec_clone, &combined_frame.rgb_jpeg).unwrap();
                                
                                // Update FPS counter
                                frame_count += 1;
                                if last_print_time.elapsed() >= FPS_PRINT_INTERVAL {
                                    let elapsed_secs = last_print_time.elapsed().as_secs_f32();
                                    let fps = frame_count as f32 / elapsed_secs;
                                    println!("Combined frame FPS: {:.2}", fps);
                                    frame_count = 0;
                                    last_print_time = Instant::now();
                                }
                            } else {
                                println!("Skipping frame due to unpacking issues");
                            }
                        }
                        Err(_) => {
                            println!("Combined stream closed");
                            break;
                        }
                    }
                }
            });

            // motion task
            let tx = raw_tx.clone();
            let motion_logger = shared_logger.clone();
            let motion_task = tokio::spawn(async move {
                loop {
                    match motion_subscriber.recv_async().await {
                        Ok(sample) => {
                            let payload = sample.payload().to_bytes().to_vec();
                            let motion_frame = MotionFrameData::decodeAndDecompress(payload);                            
                            // Send to channel
                            let frame = Frame::Motion(motion_frame.clone());
                            let _ = tx.send(frame.clone());
                            
                            // Write to log if logger exists
                            if let Some(logger) = &motion_logger {
                                // Acquire mutex lock before writing
                                let mut log_writer = logger.lock().await;
                                log_writer.write_motion(&motion_frame).unwrap();
                            }
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


