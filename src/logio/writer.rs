use std::{io::Write, fs::File, path::Path};
use crate::types::{encode_meters_to_u16, DepthFrameSerializable, Frame};

pub struct LogWriter(std::io::BufWriter<File>);

impl LogWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        Ok(Self(std::io::BufWriter::new(File::create(path)?)))
    }
    pub fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};
        match frame {
            Frame::Depth(d) => {
                self.0.write_u8(0)?;
                self.0.write_f64::<LittleEndian>(d.timestamp)?;
                let payload = DepthFrameSerializable {
                    width: d.width,
                    height: d.height,
                    timestamp: d.timestamp,
                    data: d.data.iter().map(|m| encode_meters_to_u16(*m)).collect(),
                }.encodeAndCompress();
                self.0.write_u32::<LittleEndian>(payload.len() as u32)?;
                self.0.write_all(&payload)?;
            }
            Frame::Color((jpeg, ts)) => {
                self.0.write_u8(1)?;
                self.0.write_f64::<LittleEndian>(*ts)?;
                self.0.write_u32::<LittleEndian>(jpeg.len() as u32)?;
                self.0.write_all(jpeg)?;
            }
            Frame::Motion(m) => {
                self.0.write_u8(2)?;
                self.0.write_f64::<LittleEndian>(m.timestamp)?;
                let payload = m.encodeAndCompress();
                self.0.write_u32::<LittleEndian>(payload.len() as u32)?;
                self.0.write_all(&payload)?;
            }
        }
        Ok(())
    }
}
