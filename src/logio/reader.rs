use std::{fs::File, io::Read, path::Path};
use tokio::{sync::mpsc::UnboundedSender, time::sleep};
use byteorder::{LittleEndian, ReadBytesExt};
use crate::types::{Frame, DepthFrameSerializable, MotionFrameData};

pub async fn playback(path: &Path, tx: UnboundedSender<Frame>) -> std::io::Result<()> {
    let mut f = std::io::BufReader::new(File::open(path)?);
    let mut last_ts = None;

    loop {
        let kind = match f.read_u8() { Ok(k) => k, Err(_) => break };
        let ts   = f.read_f64::<LittleEndian>()?;
        let len  = f.read_u32::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf)?;

        // pacing
        if let Some(prev) = last_ts {
            let dt = ts - prev;
            if dt > 0.0 {
                sleep(tokio::time::Duration::from_secs_f64(dt)).await;
            }
        }
        last_ts = Some(ts);

        // push to synchroniser
        let frame = match kind {
            0 => Frame::Depth(DepthFrameSerializable::decodeAndDecompress(buf)),
            1 => Frame::Color((buf, ts)),
            2 => Frame::Motion(MotionFrameData::decodeAndDecompress(buf)),
            _ => unreachable!(),
        };
        let _ = tx.send(frame);
    }
    Ok(())
}
