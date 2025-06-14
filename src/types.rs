use bincode::{Decode, Encode};
use serde::Deserialize;
use serde::Serialize;
use snap::raw::Encoder;
use turbojpeg::image::Rgb;
use zstd::decode_all;
use zstd::stream::copy_decode;
use zstd::stream::copy_encode;
use turbojpeg::Subsamp;
use turbojpeg::compress_image;
use turbojpeg::image::ImageBuffer;

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
    pub data: Vec<u16>, // distances in meters
    pub timestamp: f64,
}

#[derive(Clone)]
pub struct StampedTriple {
    pub depth: DepthFrameRealUnits,
    pub colour: (Vec<u8>, f64),        // jpeg + ts
    pub motion: MotionFrameData,
}

#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct DepthFrameRealUnits {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>, // distances in meters
    pub timestamp: f64,
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
    pub data: Vec<u8>, // RGB8
    pub timestamp: f64,
}


impl DepthFrameSerializable {
    pub fn encodeAndCompress(&self) -> Vec<u8> {
        let encoded = bincode::encode_to_vec(&self, bincode::config::standard()).unwrap();
        let mut result = Vec::new();
        copy_encode(&encoded[..], &mut result, 6).unwrap();
        result
    }

    pub fn decodeAndDecompress(encoded: Vec<u8>) -> DepthFrameRealUnits {

        let mut decompressed_data = Vec::new();
        copy_decode(&encoded[..], &mut decompressed_data).unwrap();


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




use nalgebra::{Matrix3, Vector3};

#[derive(Clone, Copy)]
pub struct Intrinsics {
    pub width: usize,
    pub height: usize,
    pub fx: f32,
    pub fy: f32,
    pub ppx: f32,
    pub ppy: f32,
}

#[derive(Clone, Copy)]
pub struct Extrinsics {
    pub rotation: Matrix3<f32>,      // row-major
    pub translation: Vector3<f32>,
}


#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct MotionFrameData {
    pub gyro: [f32; 3], // rad/s
    pub accel: [f32; 3], // m/s^2
    pub timestamp: f64, // seconds
}

impl MotionFrameData {
    pub fn new(gyro: [f32; 3], accel: [f32; 3], timestamp: f64) -> Self {
        Self { gyro, accel, timestamp }
    }

    pub fn encodeAndCompress(&self) -> Vec<u8> {
        let encoded = bincode::encode_to_vec(&self, bincode::config::standard()).unwrap();
        encoded
    }
    pub fn decodeAndDecompress(encoded: Vec<u8>) -> Self {
        let (wire, _): (MotionFrameData, _) =
        bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        wire
    }
}

/// Anything that carries a timestamp can go through the channel.
#[derive(Clone)]
pub enum Frame {
    Depth(DepthFrameRealUnits),
    Color((Vec<u8>, f64)),        // jpeg + ts   (your current return from ColorFrameSerializable)
    Motion(MotionFrameData),
}
impl Frame {
    pub fn ts(&self) -> f64 {
        match self {
            Frame::Depth(d)     => d.timestamp,
            Frame::Color((_,t)) => *t,
            Frame::Motion(m)    => m.timestamp,
        }
    }
}


#[derive(Serialize, Deserialize, Encode, Decode, Debug, Clone)]
pub struct CombinedFrameWire {
    /// JPEG-compressed RGB image
    pub rgb_jpeg: Vec<u8>,
    /// Zstd-compressed depth buffer (u16)
    pub depth_zstd: Vec<u8>,
    pub width:  u16,
    pub height: u16,
    pub timestamp: f64,          // seconds, SYSTEM_TIME domain
}

impl CombinedFrameWire {

    /// final packing for the wire
    pub fn encode(&self) -> Vec<u8> {
        let payload = bincode::encode_to_vec(self, bincode::config::standard()).unwrap();
        // a light Zstd pass mainly helps small RGB frames; level-1 keeps latency down
        let mut out = Vec::new();
        copy_encode(&payload[..], &mut out, 1).unwrap();
        out
    }

    pub fn decode(buf: &[u8]) -> Self {
        let raw = decode_all(buf).unwrap();
        let (me, _): (CombinedFrameWire, _) =
            bincode::decode_from_slice(&raw, bincode::config::standard()).unwrap();
        me
    }

    // helper to get fully-expanded data back out
    pub fn unpack(&self) -> (Vec<u8>, Vec<u16>, u16, u16, f64) {
        let rgb_raw = turbojpeg::decompress_image::<Rgb<u8>>(&self.rgb_jpeg).unwrap().into_raw();
        let depth_raw = {
            let bytes = decode_all(&self.depth_zstd[..]).unwrap();
            let (d, _): (DepthFrameSerializable, _) =
                bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            d.data
        };
        (rgb_raw, depth_raw, self.width, self.height, self.timestamp)
    }
}
