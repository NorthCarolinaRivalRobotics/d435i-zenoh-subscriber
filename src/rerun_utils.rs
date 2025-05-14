use bytemuck::cast_slice;
use rerun::{
    archetypes::DepthImage,
    datatypes::ChannelDatatype,   // <- new
    RecordingStream,
};

use crate::DepthFrameSerializable;

/// Log a floating-point depth frame (metres) to Rerun.
pub fn log_depth(rec: &RecordingStream, frame: &DepthFrameSerializable) -> anyhow::Result<()> {
    // 1. reinterpret the f32 slice as raw bytes – zero-copy:
    let raw: &[u8] = cast_slice::<f32, u8>(&frame.data);

    // 2. build the archetype (bytes, resolution, datatype):
    let depth = DepthImage::from_data_type_and_bytes(
        raw.to_vec(),                             // Vec<u8>
        [frame.width as u32, frame.height as u32],
        ChannelDatatype::F32,                     // 32-bit float depth
    )
    .with_meter(1.0);                             // the values ARE metres

    // 3. send it:
    rec.log("/camera/depth", &depth)?;
    Ok(())
}
