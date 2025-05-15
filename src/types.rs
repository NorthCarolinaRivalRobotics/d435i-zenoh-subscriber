use bincode::{Decode, Encode};
use serde::Deserialize;
use serde::Serialize;
use snap::raw::Decoder;
use snap::raw::Encoder;
use zstd::zstd_safe::compress;
use zstd::zstd_safe::decompress;
#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum ImageEncoding {
    RGB8,
    Z16,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Encode, Decode)]
pub struct RGB8 {
    b: u8,
    g: u8,
    r: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct DepthFrameSerializable {
    pub width: usize,
    pub height: usize,
    pub timestamp: f64,
    pub data: Vec<f32>, // distances in meters
}

#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
pub struct ColorFrameSerializable {
    pub width: usize,
    pub height: usize,
    pub timestamp: f64,
    pub data: Vec<RGB8>, // RGB8
}


impl DepthFrameSerializable {
    pub fn encodeAndCompress(&self) -> Vec<u8> {
        let encoded = bincode::encode_to_vec(&self, bincode::config::standard()).unwrap();
        // use snap here
        let mut encoder = Encoder::new();
        let compressed_encoded = encoder.compress_vec(&encoded).unwrap();
        compressed_encoded
    }

    pub fn decodeAndDecompress(encoded: Vec<u8>) -> Self {
        let mut decoder = Decoder::new();
        let decompressed_data = decoder.decompress_vec(encoded.as_ref()).unwrap();
        let decoded: (Self, usize) = bincode::decode_from_slice(decompressed_data.as_ref(), bincode::config::standard()).unwrap();   
        decoded.0
    }
}


impl ColorFrameSerializable {
    pub fn encodeAndCompress(&self) -> Vec<u8> {
        let encoded = bincode::encode_to_vec(&self, bincode::config::standard()).unwrap();
        // use snap here
        let mut encoder = Encoder::new();
        let compressed_encoded = encoder.compress_vec(&encoded).unwrap();
        compressed_encoded
    }
    pub fn decodeAndDecompress(encoded: Vec<u8>) -> Self {
        let mut decoder = Decoder::new();
        let decompressed_encoded = decoder.decompress_vec(&encoded).unwrap();
        let decoded: (Self, usize) = bincode::decode_from_slice(&decompressed_encoded, bincode::config::standard()).unwrap();   
        decoded.0
    }
}

