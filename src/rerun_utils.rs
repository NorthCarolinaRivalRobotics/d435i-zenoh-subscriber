use bytemuck::cast_slice;
use rerun::{
    archetypes::DepthImage, datatypes::ChannelDatatype, EncodedImage, RecordingStream
};
use crate::types::DepthFrameRealUnits;

/// Log a floating-point depth frame (metres) to Rerun.
pub fn log_depth(rec: &RecordingStream, frame: &DepthFrameRealUnits) -> anyhow::Result<()> {
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


pub fn log_aligned_depth(rec: &RecordingStream, data: &[f32], w: usize, h: usize) -> anyhow::Result<()> {
    let raw: &[u8] = bytemuck::cast_slice(data);
    let depth = DepthImage::from_data_type_and_bytes(
    raw.to_vec(), [w as u32, h as u32], ChannelDatatype::F32).with_meter(1.0);
    rec.log("/camera/depth_aligned", &depth)?;
    Ok(())
}

pub fn log_rgb_jpeg(rec: &RecordingStream, jpeg: &[u8]) -> anyhow::Result<()> {
    let img = EncodedImage::from_file_contents(jpeg.to_vec())  // alloc once
        .with_media_type("image/jpeg");                       // explicit is faster
    rec.log("/camera/rgb", &img)?;
    Ok(())
}
