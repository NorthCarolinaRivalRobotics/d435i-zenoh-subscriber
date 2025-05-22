use std::{io::Write, fs::File, path::Path};
use crate::types::{encode_meters_to_u16, DepthFrameSerializable, Frame, MotionFrameData};

pub struct LogWriter(std::io::BufWriter<File>);
const KIND_COMBINED: u8 = 0;
const KIND_MOTION  : u8 = 1;

impl LogWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self(std::io::BufWriter::new(File::create(path)?)))
    }
        /// Write an already-encoded CombinedFrameWire
        pub fn write_combined(&mut self, ts: f64, payload: &[u8]) -> std::io::Result<()> {
            use byteorder::{LittleEndian, WriteBytesExt};
            self.0.write_u8(KIND_COMBINED)?;
            self.0.write_f64::<LittleEndian>(ts)?;
            self.0.write_u32::<LittleEndian>(payload.len() as u32)?;
            self.0.write_all(payload)
        }
    
        /// Motion unchanged — you rarely record that synchronously with colour/depth
        pub fn write_motion(&mut self, m: &MotionFrameData) -> std::io::Result<()> {
            use byteorder::{LittleEndian, WriteBytesExt};
            self.0.write_u8(KIND_MOTION)?;
            self.0.write_f64::<LittleEndian>(m.timestamp)?;
            let blob = m.encodeAndCompress();
            self.0.write_u32::<LittleEndian>(blob.len() as u32)?;
            self.0.write_all(&blob)
        }
    
}
