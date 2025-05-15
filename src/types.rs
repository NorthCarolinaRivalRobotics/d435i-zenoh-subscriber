use bincode::{Decode, Encode};
use serde::Deserialize;
use serde::Serialize;
use snap::raw::Decoder;
use snap::raw::Encoder;
use turbojpeg::decompress_image;
use turbojpeg::image::Rgb;
use zstd::zstd_safe::compress;
use zstd::zstd_safe::decompress;
use turbojpeg::Subsamp;
use turbojpeg::compress_image;
use turbojpeg::image::ImageBuffer;

use crate::rerun_utils::log_rgb_jpeg;
#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum ImageEncoding {
    RGB8,
    Z16,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Encode, Decode)]
pub struct RGB8Local {
    b: u8,
    g: u8,
    r: u8,
}

const DEPTH_SCALE_FACTOR: u16 = 8738; // multiply by this to convert meters to u16
const MINIMUM_DISTANCE_METERS: f32 = 0.5;

pub fn encode_meters_to_u16(meters: f32)     -> u16 {
    ((meters - MINIMUM_DISTANCE_METERS) * DEPTH_SCALE_FACTOR as f32) as u16
}

pub fn decode_u16_to_meters(code: u16) -> f32 {
    (code as f32) / DEPTH_SCALE_FACTOR as f32 + MINIMUM_DISTANCE_METERS
}


#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct DepthFrameSerializable {
    pub width: usize,
    pub height: usize,
    pub timestamp: f64,
    pub data: Vec<u16>, // distances in meters
}

pub struct DepthFrameRealUnits {
    pub width: usize,
    pub height: usize,
    pub timestamp: f64,
    pub data: Vec<f32>, // distances in meters
}

#[derive(Encode, Decode)]
pub struct ImageForWire {
    pub image: Vec<u8>,
    pub timestamp: f64,
}


#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct ColorFrameSerializable {
    pub width: usize,
    pub height: usize,
    pub timestamp: f64,
    pub data: Vec<u8>, // RGB8
}


impl DepthFrameSerializable {
    pub fn encodeAndCompress(&self) -> Vec<u8> {
        let encoded = bincode::encode_to_vec(&self, bincode::config::standard()).unwrap();
        // use snap here
        let mut encoder = Encoder::new();
        let compressed_encoded = encoder.compress_vec(&encoded).unwrap();
        compressed_encoded
    }

    pub fn decodeAndDecompress(encoded: Vec<u8>) -> DepthFrameRealUnits {
        let mut decoder = Decoder::new();
        let decompressed_data = decoder.decompress_vec(encoded.as_ref()).unwrap();
        let decoded: (Self, usize) = bincode::decode_from_slice(decompressed_data.as_ref(), bincode::config::standard()).unwrap();   
        let mut real_units = DepthFrameRealUnits {
            width: decoded.0.width,
            height: decoded.0.height,
            timestamp: decoded.0.timestamp,
            data: decoded.0.data.iter().map(|&code| decode_u16_to_meters(code)).collect(),
        };
        real_units
    }
}


impl ColorFrameSerializable {
    pub fn encodeAndCompress(&self) -> Vec<u8> {
        let jpeg = compress_image::<Rgb<u8>>(&ImageBuffer::from_vec(self.width as u32, self.height as u32, self.data.clone()).unwrap(), 75, Subsamp::Sub4x1).unwrap();
        let new_struct: ImageForWire = ImageForWire {
            image: jpeg.to_vec(),
            timestamp: self.timestamp,
        };
        let encoded = bincode::encode_to_vec(&new_struct, bincode::config::standard()).unwrap();
        // use snap here
        let mut encoder = Encoder::new();
        let compressed_encoded = encoder.compress_vec(&encoded).unwrap();
        compressed_encoded
    }
    pub fn decodeAndDecompress(encoded: Vec<u8>) -> (Vec<u8>, f64) {
        let (wire, _): (ImageForWire, _) =
        bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

        // JPEG --------------------------------------------------------------
        debug_assert!(wire.image.starts_with(&[0xFF, 0xD8]));
        // let rgb = turbojpeg::decompress_image::<Rgb<u8>>(&wire.image).unwrap();

        (wire.image, wire.timestamp)

    }
    
}

