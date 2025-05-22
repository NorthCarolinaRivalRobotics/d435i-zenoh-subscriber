use std::{fs::File, io::Read, path::Path, time::Instant};
use rerun::{dataframe::TimelineName, TimeCell};
use tokio::{sync::mpsc::UnboundedSender, time::sleep};
use byteorder::{LittleEndian, ReadBytesExt, LE};
use crate::{rerun_utils::{log_aligned_depth, log_rgb_jpeg}, types::{CombinedFrameWire, DepthFrameRealUnits, DepthFrameSerializable, Frame, MotionFrameData}};

/// Sleep until `desired_since_start` seconds have really passed since `wall_start`.
async fn pace(desired_since_start: f64, wall_start: &Instant) {
    let actual = wall_start.elapsed().as_secs_f64();
    if desired_since_start > actual {
        tokio::time::sleep(std::time::Duration::from_secs_f64(
            desired_since_start - actual,
        ))
        .await;
    }
}

pub fn decode_u16_to_meters(val: u16) -> f32 {
    val as f32 / 1000.0 // Assuming depth values are in millimeters
}

pub async fn playback(path: &Path, tx: UnboundedSender<Frame>) -> std::io::Result<()> {
    // create a Rerun recording just for playback visualisation
    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();

    const KIND_COMBINED: u8 = 0;
    const KIND_MOTION: u8 = 1;

    let mut f = std::io::BufReader::new(File::open(path).unwrap());
    let mut start_ts: Option<f64> = None;
    let mut wall_start: Option<Instant> = None;

    loop {
        // read the header
        let kind   = match f.read_u8() { Ok(k) => k, Err(_) => break };
        // ‼️ correct order: timestamp THEN length
        let ts_ms  = f.read_f64::<LittleEndian>()?;           // <- 8 bytes
        let len    = f.read_u32::<LittleEndian>()? as usize;  // <- 4 bytes
        let ts    = ts_ms / 1_000.0;
        println!("kind={kind}, ts={ts}, len={len}");

        if start_ts.is_none() {
            start_ts = Some(ts);
            wall_start = Some(Instant::now());
        }

        let desired = ts - start_ts.unwrap();
        pace(desired, &wall_start.as_ref().unwrap()).await;
        
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf).unwrap();

        let time = ts - start_ts.unwrap();
        println!("reader: time {}", time);

        // rebuild the frame *in seconds*
        match kind {
            KIND_COMBINED => {
                let frame_wire = CombinedFrameWire::decode(&buf);
                let (rgb, depth, w, h, ts) = frame_wire.unpack();

                // push into the live pipeline exactly as the network path does
                let depth_ru = DepthFrameRealUnits {
                    width: w as usize,
                    height: h as usize,
                    timestamp: ts,
                    data: depth.iter().map(|c| decode_u16_to_meters(*c)).collect(),
                };
                log_aligned_depth(&rec, &depth_ru.data, w as usize, h as usize).ok();
                log_rgb_jpeg(&rec, &frame_wire.rgb_jpeg).ok();

                let _ = tx.send(Frame::Depth(depth_ru.clone()));
                let _ = tx.send(Frame::Color((frame_wire.rgb_jpeg.clone(), ts)));
            }

            KIND_MOTION => {
                let m = MotionFrameData::decodeAndDecompress(buf);
                let _ = tx.send(Frame::Motion(m));
            }

            _ => unreachable!(),
        }
        
        println!("reader: pushed frame");
    }
    Ok(())
}
