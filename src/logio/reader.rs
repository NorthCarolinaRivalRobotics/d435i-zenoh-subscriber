use std::{fs::File, io::Read, path::Path, time::Instant};
use rerun::{dataframe::TimelineName, TimeCell};
use tokio::{sync::mpsc::UnboundedSender, time::sleep};
use byteorder::{LittleEndian, ReadBytesExt};
use crate::{rerun_utils::{log_aligned_depth, log_rgb_jpeg}, types::{DepthFrameSerializable, Frame, MotionFrameData}};

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

pub async fn playback(path: &Path, tx: UnboundedSender<Frame>) -> std::io::Result<()> {
    // create a Rerun recording just for playback visualisation
    let rec = rerun::RecordingStreamBuilder::new("d435i_playback").spawn().unwrap();

    let mut f = std::io::BufReader::new(File::open(path)?);
    let mut start_ts: Option<f64> = None;
    let mut wall_start: Option<Instant> = None;

    loop {
        let kind = match f.read_u8() { Ok(k) => k, Err(_) => break };
        let ts   = f.read_f64::<LittleEndian>()? ; // validated with printing that this is in seconds 
        let ts_ns = (ts * 1e9) as i64;
        
        let ts_sec = ts;          // already seconds
        if start_ts.is_none() {
            start_ts = Some(ts_sec);
            wall_start = Some(Instant::now());
        }

        let desired = ts_sec - start_ts.unwrap();
        pace(desired, &wall_start.as_ref().unwrap()).await;
        
        let len  = f.read_u32::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf)?;

        let time = ts_ns - (start_ts.unwrap() * 1e9) as i64;
        println!("reader: time {}", time as f64 / 1e9 as f64);
        rec.set_duration_secs(TimelineName::log_time(), ts);

        // rebuild the frame *in seconds*
        let frame = match kind {
            0 => {
                let mut d = DepthFrameSerializable::decodeAndDecompress(buf);
                d.timestamp = ts;
                // log for rerun
                log_aligned_depth(&rec, &d.data, d.width, d.height).ok();
                Frame::Depth(d)
            }
            1 => {
                log_rgb_jpeg(&rec, &buf).ok();
                Frame::Color((buf, ts))
            }
            2 => {
                let mut m = MotionFrameData::decodeAndDecompress(buf);
                m.timestamp = ts;
                Frame::Motion(m)
            }
            _ => unreachable!(),
        };

        // **send to the synchroniser's channel**
        let _ = tx.send(frame);
        println!("reader: pushed frame");
    }
    Ok(())
}
