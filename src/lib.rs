use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyRuntimeError;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use std::time::{Duration, Instant};
use nalgebra::{Matrix3, Vector3};
use numpy::{PyArray1, PyArray2, IntoPyArray};

mod types;
use types::*;

/// Python wrapper for camera intrinsics
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyIntrinsics {
    pub width: usize,
    pub height: usize,
    pub fx: f32,
    pub fy: f32,
    pub ppx: f32,
    pub ppy: f32,
}

#[pymethods]
impl PyIntrinsics {
    #[new]
    fn new(width: usize, height: usize, fx: f32, fy: f32, ppx: f32, ppy: f32) -> Self {
        Self { width, height, fx, fy, ppx, ppy }
    }
    
    fn __repr__(&self) -> String {
        format!("PyIntrinsics({}x{}, fx={:.2}, fy={:.2}, ppx={:.2}, ppy={:.2})", 
                self.width, self.height, self.fx, self.fy, self.ppx, self.ppy)
    }
}

/// Python wrapper for camera extrinsics
#[pyclass]
#[derive(Clone)]
pub struct PyExtrinsics {
    pub rotation: Matrix3<f32>,
    pub translation: Vector3<f32>,
}

#[pymethods]
impl PyExtrinsics {
    #[new]
    fn new(rotation: Vec<f32>, translation: Vec<f32>) -> PyResult<Self> {
        if rotation.len() != 9 {
            return Err(PyRuntimeError::new_err("Rotation matrix must have 9 elements"));
        }
        if translation.len() != 3 {
            return Err(PyRuntimeError::new_err("Translation vector must have 3 elements"));
        }
        
        let rot_matrix = Matrix3::from_row_slice(&rotation);
        let trans_vector = Vector3::new(translation[0], translation[1], translation[2]);
        
        Ok(Self {
            rotation: rot_matrix,
            translation: trans_vector,
        })
    }
}

/// Python wrapper for motion data
#[pyclass]
#[derive(Clone)]
pub struct PyMotionFrame {
    #[pyo3(get)]
    pub gyro: Vec<f32>,
    #[pyo3(get)]
    pub accel: Vec<f32>,
    #[pyo3(get)]
    pub timestamp: f64,
}

#[pymethods]
impl PyMotionFrame {
    fn __repr__(&self) -> String {
        format!("PyMotionFrame(gyro={:?}, accel={:?}, timestamp={:.3})", 
                self.gyro, self.accel, self.timestamp)
    }
}

/// Python wrapper for RGB frame
#[pyclass]
#[derive(Clone)]
pub struct PyRgbFrame {
    pub data: Vec<u8>,
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub width: u16,
    #[pyo3(get)]
    pub height: u16,
}

#[pymethods]
impl PyRgbFrame {
    fn get_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.data)
    }
    
    fn __repr__(&self) -> String {
        format!("PyRgbFrame({}x{}, {} bytes, timestamp={:.3})", 
                self.width, self.height, self.data.len(), self.timestamp)
    }
}

/// Python wrapper for depth frame
#[pyclass]
#[derive(Clone)]
pub struct PyDepthFrame {
    pub data: Vec<f32>,
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
}

#[pymethods]
impl PyDepthFrame {
    fn get_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec_bound(py, self.data.clone())
    }
    
    fn get_data_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let array = ndarray::Array2::from_shape_vec((self.height, self.width), self.data.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;
        Ok(array.into_pyarray_bound(py))
    }
    
    fn __repr__(&self) -> String {
        format!("PyDepthFrame({}x{}, {} points, timestamp={:.3})", 
                self.width, self.height, self.data.len(), self.timestamp)
    }
}

/// Combined frame data for easy access
#[pyclass]
#[derive(Clone)]
pub struct PyFrameData {
    #[pyo3(get)]
    pub rgb: Option<PyRgbFrame>,
    #[pyo3(get)]
    pub depth: Option<PyDepthFrame>,
    #[pyo3(get)]
    pub motion: Option<PyMotionFrame>,
    #[pyo3(get)]
    pub frame_count: u64,
}

#[pymethods]
impl PyFrameData {
    fn __repr__(&self) -> String {
        let rgb_info = if let Some(ref rgb) = self.rgb {
            format!("RGB({}x{})", rgb.width, rgb.height)
        } else {
            "RGB(None)".to_string()
        };
        
        let depth_info = if let Some(ref depth) = self.depth {
            format!("Depth({}x{})", depth.width, depth.height)
        } else {
            "Depth(None)".to_string()
        };
        
        let motion_info = if self.motion.is_some() {
            "Motion(available)"
        } else {
            "Motion(None)"
        };
        
        format!("PyFrameData({}, {}, {}, frame={})", rgb_info, depth_info, motion_info, self.frame_count)
    }
}

/// Main Zenoh subscriber class with simple polling API
#[pyclass]
pub struct ZenohD435iSubscriber {
    runtime: Arc<Runtime>,
    session: Option<Arc<zenoh::Session>>,
    latest_rgb: Arc<Mutex<Option<PyRgbFrame>>>,
    latest_depth: Arc<Mutex<Option<PyDepthFrame>>>,
    latest_motion: Arc<Mutex<Option<PyMotionFrame>>>,
    frame_count: Arc<Mutex<u64>>,
    running: Arc<Mutex<bool>>,
    depth_intrinsics: Option<PyIntrinsics>,
    color_intrinsics: Option<PyIntrinsics>,
    extrinsics: Option<PyExtrinsics>,
}

#[pymethods]
impl ZenohD435iSubscriber {
    #[new]
    fn new() -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
        
        Ok(Self {
            runtime,
            session: None,
            latest_rgb: Arc::new(Mutex::new(None)),
            latest_depth: Arc::new(Mutex::new(None)),
            latest_motion: Arc::new(Mutex::new(None)),
            frame_count: Arc::new(Mutex::new(0)),
            running: Arc::new(Mutex::new(false)),
            depth_intrinsics: None,
            color_intrinsics: None,
            extrinsics: None,
        })
    }
    
    /// Connect to Zenoh
    fn connect(&mut self) -> PyResult<()> {
        let session = self.runtime.block_on(async {
            zenoh::open(zenoh::Config::default()).await
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        self.session = Some(Arc::new(session));
        Ok(())
    }
    
    /// Set camera intrinsics
    fn set_depth_intrinsics(&mut self, intrinsics: PyIntrinsics) {
        self.depth_intrinsics = Some(intrinsics);
    }
    
    fn set_color_intrinsics(&mut self, intrinsics: PyIntrinsics) {
        self.color_intrinsics = Some(intrinsics);
    }
    
    fn set_extrinsics(&mut self, extrinsics: PyExtrinsics) {
        self.extrinsics = Some(extrinsics);
    }
    
    /// Start subscribing to Zenoh topics
    fn start_subscribing(&mut self) -> PyResult<()> {
        let session = self.session.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Not connected to Zenoh. Call connect() first."))?
            .clone();
        
        let latest_rgb = self.latest_rgb.clone();
        let latest_depth = self.latest_depth.clone();
        let latest_motion = self.latest_motion.clone();
        let frame_count = self.frame_count.clone();
        let running = self.running.clone();
        
        *running.lock().unwrap() = true;
        
        let runtime = self.runtime.clone();
        
        // Spawn the subscriber tasks
        runtime.spawn(async move {
            let combined_subscriber = session.declare_subscriber("camera/combined").await.unwrap();
            let motion_subscriber = session.declare_subscriber("camera/motion").await.unwrap();
            
            // Combined frame task
            let running_combined = running.clone();
            let rgb_store = latest_rgb.clone();
            let depth_store = latest_depth.clone();
            let frame_counter = frame_count.clone();
            let combined_task = tokio::spawn(async move {
                let mut last_print_time = Instant::now();
                let mut local_frame_count = 0u32;
                const FPS_PRINT_INTERVAL: Duration = Duration::from_secs(5);
                
                println!("Combined frame subscriber started, listening on 'camera/combined'");
                
                while *running_combined.lock().unwrap() {
                    match combined_subscriber.recv_async().await {
                        Ok(sample) => {
                            let payload = sample.payload().to_bytes().to_vec();
                            let combined_frame = CombinedFrameWire::decode(&payload);
                            
                            if let Some((rgb_raw, depth_raw, width, height, timestamp)) = unpack_combined_frame(&combined_frame) {
                                // Create frames
                                let rgb_frame = PyRgbFrame {
                                    data: rgb_raw,
                                    timestamp,
                                    width,
                                    height,
                                };
                                
                                let depth_frame = PyDepthFrame {
                                    width: width as usize,
                                    height: height as usize,
                                    timestamp,
                                    data: depth_raw.iter().map(|&code| decode_u16_to_meters(code)).collect(),
                                };
                                
                                // Store latest frames
                                if let Ok(mut rgb_lock) = rgb_store.lock() {
                                    *rgb_lock = Some(rgb_frame);
                                }
                                if let Ok(mut depth_lock) = depth_store.lock() {
                                    *depth_lock = Some(depth_frame);
                                }
                                if let Ok(mut count_lock) = frame_counter.lock() {
                                    *count_lock += 1;
                                }
                                
                                // Update FPS counter
                                local_frame_count += 1;
                                if last_print_time.elapsed() >= FPS_PRINT_INTERVAL {
                                    let elapsed_secs = last_print_time.elapsed().as_secs_f32();
                                    let fps = local_frame_count as f32 / elapsed_secs;
                                    println!("Combined frame FPS: {:.2}", fps);
                                    local_frame_count = 0;
                                    last_print_time = Instant::now();
                                }
                            } else {
                                println!("Failed to unpack combined frame");
                            }
                        }
                        Err(e) => {
                            println!("Combined subscriber error: {:?}", e);
                            break;
                        }
                    }
                }
                println!("Combined frame subscriber stopped");
            });
            
            // Motion task
            let running_motion = running.clone();
            let motion_store = latest_motion.clone();
            let motion_task = tokio::spawn(async move {
                println!("Motion subscriber started, listening on 'camera/motion'");
                
                while *running_motion.lock().unwrap() {
                    match motion_subscriber.recv_async().await {
                        Ok(sample) => {
                            let payload = sample.payload().to_bytes().to_vec();
                            let motion_frame = MotionFrameData::decodeAndDecompress(payload);
                            
                            let py_motion = PyMotionFrame {
                                gyro: motion_frame.gyro.to_vec(),
                                accel: motion_frame.accel.to_vec(),
                                timestamp: motion_frame.timestamp,
                            };
                            
                            // Store latest motion data
                            if let Ok(mut motion_lock) = motion_store.lock() {
                                *motion_lock = Some(py_motion);
                            }
                        }
                        Err(e) => {
                            println!("Motion subscriber error: {:?}", e);
                            break;
                        }
                    }
                }
                println!("Motion subscriber stopped");
            });
            
            // Wait for both tasks
            let _ = tokio::join!(combined_task, motion_task);
        });
        
        Ok(())
    }
    
    /// Get the latest frame data (non-blocking)
    fn get_latest_frames(&self) -> PyFrameData {
        let rgb = if let Ok(rgb_lock) = self.latest_rgb.lock() {
            rgb_lock.clone()
        } else {
            None
        };
        
        let depth = if let Ok(depth_lock) = self.latest_depth.lock() {
            depth_lock.clone()
        } else {
            None
        };
        
        let motion = if let Ok(motion_lock) = self.latest_motion.lock() {
            motion_lock.clone()
        } else {
            None
        };
        
        let frame_count = if let Ok(count_lock) = self.frame_count.lock() {
            *count_lock
        } else {
            0
        };
        
        PyFrameData {
            rgb,
            depth,
            motion,
            frame_count,
        }
    }
    
    /// Check if any new data is available
    fn has_new_data(&self) -> bool {
        let frame_count = if let Ok(count_lock) = self.frame_count.lock() {
            *count_lock
        } else {
            0
        };
        frame_count > 0
    }
    
    /// Get frame statistics
    fn get_stats(&self) -> String {
        let frame_count = if let Ok(count_lock) = self.frame_count.lock() {
            *count_lock
        } else {
            0
        };
        
        let has_rgb = if let Ok(rgb_lock) = self.latest_rgb.lock() {
            rgb_lock.is_some()
        } else {
            false
        };
        
        let has_depth = if let Ok(depth_lock) = self.latest_depth.lock() {
            depth_lock.is_some()
        } else {
            false
        };
        
        let has_motion = if let Ok(motion_lock) = self.latest_motion.lock() {
            motion_lock.is_some()
        } else {
            false
        };
        
        format!("Frames received: {}, RGB: {}, Depth: {}, Motion: {}", 
                frame_count, has_rgb, has_depth, has_motion)
    }
    
    /// Stop subscribing
    fn stop(&mut self) {
        *self.running.lock().unwrap() = false;
    }
    
    /// Check if currently running
    fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

/// Unpacks a combined frame into RGB and depth data
fn unpack_combined_frame(frame: &CombinedFrameWire) -> Option<(Vec<u8>, Vec<u16>, u16, u16, f64)> {
    // 1. RGB decompression
    let rgb_raw = match turbojpeg::decompress_image::<turbojpeg::image::Rgb<u8>>(&frame.rgb_jpeg) {
        Ok(img) => img.into_raw(),
        Err(e) => {
            println!("RGB decompression error: {:?}", e);
            return None;
        }
    };
    
    // 2. Depth decompression and deserialization
    let depth_raw = match zstd::decode_all(&frame.depth_zstd[..]) {
        Ok(decompressed) => {
            match bincode::decode_from_slice::<DepthFrameSerializable, _>(&decompressed, bincode::config::standard()) {
                Ok((d, _)) => {
                    if d.data.is_empty() {
                        println!("Warning: Depth data contains 0 points");
                        return None;
                    }
                    d.data
                },
                Err(e) => {
                    println!("Depth deserialization error: {:?}", e);
                    return None;
                }
            }
        },
        Err(e) => {
            println!("Depth decompression error: {:?}", e);
            return None;
        }
    };
    
    Some((rgb_raw, depth_raw, frame.width, frame.height, frame.timestamp))
}

/// Python module definition
#[pymodule]
fn zenoh_d435i_subscriber(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZenohD435iSubscriber>()?;
    m.add_class::<PyIntrinsics>()?;
    m.add_class::<PyExtrinsics>()?;
    m.add_class::<PyMotionFrame>()?;
    m.add_class::<PyRgbFrame>()?;
    m.add_class::<PyDepthFrame>()?;
    m.add_class::<PyFrameData>()?;
    Ok(())
} 