use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyRuntimeError;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use std::time::{Duration, Instant};
use nalgebra::{Matrix3, Vector3};
use numpy::{PyArray1, PyArray2, IntoPyArray};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

mod types;
use types::*;

// Constants for latency optimization
const MAX_QUEUED_FRAMES: usize = 2;  // Drop frames if queue gets too long
const PROCESSING_TIMEOUT_MS: u64 = 50;  // Max time to spend on one frame

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

/// Python wrapper for depth frame with lazy conversion
#[pyclass]
#[derive(Clone)]
pub struct PyDepthFrame {
    pub raw_data: Vec<u16>,  // Store raw u16 data
    converted_data: Option<Vec<f32>>,  // Cache converted data
    #[pyo3(get)]
    pub timestamp: f64,
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
}

impl PyDepthFrame {
    fn new(raw_data: Vec<u16>, timestamp: f64, width: usize, height: usize) -> Self {
        Self {
            raw_data,
            converted_data: None,
            timestamp,
            width,
            height,
        }
    }
    
    fn ensure_converted(&mut self) {
        if self.converted_data.is_none() {
            self.converted_data = Some(
                self.raw_data.iter().map(|&code| decode_u16_to_meters(code)).collect()
            );
        }
    }
}

#[pymethods]
impl PyDepthFrame {
    fn get_data<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.ensure_converted();
        PyArray1::from_vec_bound(py, self.converted_data.as_ref().unwrap().clone())
    }
    
    fn get_data_2d<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.ensure_converted();
        let data = self.converted_data.as_ref().unwrap();
        let array = ndarray::Array2::from_shape_vec((self.height, self.width), data.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;
        Ok(array.into_pyarray_bound(py))
    }
    
    fn get_raw_data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        PyArray1::from_vec_bound(py, self.raw_data.clone())
    }
    
    fn __repr__(&self) -> String {
        format!("PyDepthFrame({}x{}, {} points, timestamp={:.3})", 
                self.width, self.height, self.raw_data.len(), self.timestamp)
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

/// Frame processing statistics
struct ProcessingStats {
    frames_received: AtomicU64,
    frames_dropped: AtomicU64,
    frames_processed: AtomicU64,
    processing_queue_len: AtomicU64,
}

impl ProcessingStats {
    fn new() -> Self {
        Self {
            frames_received: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            frames_processed: AtomicU64::new(0),
            processing_queue_len: AtomicU64::new(0),
        }
    }
}

/// Main Zenoh subscriber class with optimized low-latency processing
#[pyclass]
pub struct ZenohD435iSubscriber {
    runtime: Arc<Runtime>,
    session: Option<Arc<zenoh::Session>>,
    latest_rgb: Arc<Mutex<Option<PyRgbFrame>>>,
    latest_depth: Arc<Mutex<Option<PyDepthFrame>>>,
    latest_motion: Arc<Mutex<Option<PyMotionFrame>>>,
    frame_count: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    processing_stats: Arc<ProcessingStats>,
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
            frame_count: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(false)),
            processing_stats: Arc::new(ProcessingStats::new()),
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
    
    /// Start subscribing to Zenoh topics with optimized processing
    fn start_subscribing(&mut self) -> PyResult<()> {
        let session = self.session.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Not connected to Zenoh. Call connect() first."))?
            .clone();
        
        let latest_rgb = self.latest_rgb.clone();
        let latest_depth = self.latest_depth.clone();
        let latest_motion = self.latest_motion.clone();
        let frame_count = self.frame_count.clone();
        let running = self.running.clone();
        let processing_stats = self.processing_stats.clone();
        
        running.store(true, Ordering::Relaxed);
        
        let runtime = self.runtime.clone();
        
        // Spawn the optimized subscriber tasks
        runtime.spawn(async move {
            let combined_subscriber = session.declare_subscriber("camera/combined").await.unwrap();
            let motion_subscriber = session.declare_subscriber("camera/motion").await.unwrap();
            
            // Combined frame task with frame dropping and async processing
            let running_combined = running.clone();
            let rgb_store = latest_rgb.clone();
            let depth_store = latest_depth.clone();
            let frame_counter = frame_count.clone();
            let stats = processing_stats.clone();
            
            let combined_task = tokio::spawn(async move {
                let mut last_print_time = Instant::now();
                let mut local_frame_count = 0u32;
                let mut pending_frames = 0usize;
                const FPS_PRINT_INTERVAL: Duration = Duration::from_secs(5);
                
                println!("Optimized combined frame subscriber started, listening on 'camera/combined'");
                
                while running_combined.load(Ordering::Relaxed) {
                    match combined_subscriber.recv_async().await {
                        Ok(sample) => {
                            stats.frames_received.fetch_add(1, Ordering::Relaxed);
                            
                            // Frame dropping logic - if we're behind, drop frames
                            if pending_frames >= MAX_QUEUED_FRAMES {
                                stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            
                            let payload = sample.payload().to_bytes().to_vec();
                            pending_frames += 1;
                            stats.processing_queue_len.store(pending_frames as u64, Ordering::Relaxed);
                            
                            // Move heavy processing to blocking thread pool
                            let rgb_store_clone = rgb_store.clone();
                            let depth_store_clone = depth_store.clone();
                            let frame_counter_clone = frame_counter.clone();
                            let stats_clone = stats.clone();
                            
                            tokio::task::spawn_blocking(move || {
                                let process_start = Instant::now();
                                
                                // Decode frame in blocking context
                                let combined_frame = CombinedFrameWire::decode(&payload);
                                
                                if let Some((rgb_raw, depth_raw, width, height, timestamp)) = unpack_combined_frame(&combined_frame) {
                                    // Check processing timeout
                                    if process_start.elapsed().as_millis() > PROCESSING_TIMEOUT_MS as u128 {
                                        println!("Warning: Frame processing exceeded timeout");
                                        return;
                                    }
                                    
                                    // Create frames
                                    let rgb_frame = PyRgbFrame {
                                        data: rgb_raw,
                                        timestamp,
                                        width,
                                        height,
                                    };
                                    
                                    // Use lazy depth conversion
                                    let depth_frame = PyDepthFrame::new(
                                        depth_raw, 
                                        timestamp, 
                                        width as usize, 
                                        height as usize
                                    );
                                    
                                    // Update stores with combined lock to reduce contention
                                    let mut frames_updated = false;
                                    if let (Ok(mut rgb_lock), Ok(mut depth_lock)) = 
                                        (rgb_store_clone.try_lock(), depth_store_clone.try_lock()) {
                                        *rgb_lock = Some(rgb_frame);
                                        *depth_lock = Some(depth_frame);
                                        frames_updated = true;
                                    }
                                    
                                    if frames_updated {
                                        frame_counter_clone.fetch_add(1, Ordering::Relaxed);
                                        stats_clone.frames_processed.fetch_add(1, Ordering::Relaxed);
                                    }
                                } else {
                                    println!("Failed to unpack combined frame");
                                }
                            });
                            
                            pending_frames = pending_frames.saturating_sub(1);
                            
                            // Update FPS counter
                            local_frame_count += 1;
                            if last_print_time.elapsed() >= FPS_PRINT_INTERVAL {
                                let elapsed_secs = last_print_time.elapsed().as_secs_f32();
                                let fps = local_frame_count as f32 / elapsed_secs;
                                let total_received = stats.frames_received.load(Ordering::Relaxed);
                                let total_dropped = stats.frames_dropped.load(Ordering::Relaxed);
                                let total_processed = stats.frames_processed.load(Ordering::Relaxed);
                                
                                println!(
                                    "Combined frame FPS: {:.2} | Received: {} | Processed: {} | Dropped: {} ({:.1}%)",
                                    fps, total_received, total_processed, total_dropped,
                                    if total_received > 0 { (total_dropped as f32 / total_received as f32) * 100.0 } else { 0.0 }
                                );
                                local_frame_count = 0;
                                last_print_time = Instant::now();
                            }
                        }
                        Err(e) => {
                            println!("Combined subscriber error: {:?}", e);
                            break;
                        }
                    }
                }
                println!("Optimized combined frame subscriber stopped");
            });
            
            // Motion task (unchanged as it's already lightweight)
            let running_motion = running.clone();
            let motion_store = latest_motion.clone();
            let motion_task = tokio::spawn(async move {
                println!("Motion subscriber started, listening on 'camera/motion'");
                
                while running_motion.load(Ordering::Relaxed) {
                    match motion_subscriber.recv_async().await {
                        Ok(sample) => {
                            let payload = sample.payload().to_bytes().to_vec();
                            
                            // Motion processing is lightweight, keep it in async context
                            let motion_frame = MotionFrameData::decodeAndDecompress(payload);
                            
                            let py_motion = PyMotionFrame {
                                gyro: motion_frame.gyro.to_vec(),
                                accel: motion_frame.accel.to_vec(),
                                timestamp: motion_frame.timestamp,
                            };
                            
                            // Store latest motion data
                            if let Ok(mut motion_lock) = motion_store.try_lock() {
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
    
    /// Get the latest frame data (non-blocking with try_lock)
    fn get_latest_frames(&self) -> PyFrameData {
        let rgb = if let Ok(rgb_lock) = self.latest_rgb.try_lock() {
            rgb_lock.clone()
        } else {
            None
        };
        
        let depth = if let Ok(depth_lock) = self.latest_depth.try_lock() {
            depth_lock.clone()
        } else {
            None
        };
        
        let motion = if let Ok(motion_lock) = self.latest_motion.try_lock() {
            motion_lock.clone()
        } else {
            None
        };
        
        let frame_count = self.frame_count.load(Ordering::Relaxed);
        
        PyFrameData {
            rgb,
            depth,
            motion,
            frame_count,
        }
    }
    
    /// Check if any new data is available
    fn has_new_data(&self) -> bool {
        self.frame_count.load(Ordering::Relaxed) > 0
    }
    
    /// Get frame statistics including latency metrics
    fn get_stats(&self) -> String {
        let frame_count = self.frame_count.load(Ordering::Relaxed);
        let frames_received = self.processing_stats.frames_received.load(Ordering::Relaxed);
        let frames_dropped = self.processing_stats.frames_dropped.load(Ordering::Relaxed);
        let frames_processed = self.processing_stats.frames_processed.load(Ordering::Relaxed);
        let queue_len = self.processing_stats.processing_queue_len.load(Ordering::Relaxed);
        
        let has_rgb = if let Ok(rgb_lock) = self.latest_rgb.try_lock() {
            rgb_lock.is_some()
        } else {
            false
        };
        
        let has_depth = if let Ok(depth_lock) = self.latest_depth.try_lock() {
            depth_lock.is_some()
        } else {
            false
        };
        
        let has_motion = if let Ok(motion_lock) = self.latest_motion.try_lock() {
            motion_lock.is_some()
        } else {
            false
        };
        
        let drop_rate = if frames_received > 0 {
            (frames_dropped as f32 / frames_received as f32) * 100.0
        } else {
            0.0
        };
        
        format!(
            "Frames: {} | Received: {} | Processed: {} | Dropped: {} ({:.1}%) | Queue: {} | RGB: {} | Depth: {} | Motion: {}", 
            frame_count, frames_received, frames_processed, frames_dropped, drop_rate, queue_len, has_rgb, has_depth, has_motion
        )
    }
    
    /// Stop subscribing
    fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
    
    /// Check if currently running
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
    
    /// Reset statistics
    fn reset_stats(&self) {
        self.processing_stats.frames_received.store(0, Ordering::Relaxed);
        self.processing_stats.frames_dropped.store(0, Ordering::Relaxed);
        self.processing_stats.frames_processed.store(0, Ordering::Relaxed);
        self.processing_stats.processing_queue_len.store(0, Ordering::Relaxed);
        self.frame_count.store(0, Ordering::Relaxed);
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