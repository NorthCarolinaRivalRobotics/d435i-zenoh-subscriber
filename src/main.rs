use rerun_utils::log_depth;
use types::{ColorFrameSerializable, DepthFrameSerializable};

mod types;
mod rerun_utils;

#[tokio::main]
async fn main() {
    let session = zenoh::open(zenoh::Config::default()).await.unwrap();
    let rec = rerun::RecordingStreamBuilder::new("d435i").spawn().unwrap();
    let depthSubscriber = session.declare_subscriber("camera/depth").await.unwrap();
    let colorSubscriber = session.declare_subscriber("camera/rgb").await.unwrap();
    loop {
        tokio::select! {
            sample = depthSubscriber.recv_async() => {
                if let Ok(sample) = sample {
                    let depth_frame: DepthFrameSerializable = bincode::decode_from_slice(sample.payload().to_bytes().as_ref(), bincode::config::standard()).unwrap().0;
                    println!("Received (depth): {:?}", depth_frame.data.len());
                    log_depth(&rec, &depth_frame).unwrap();
                } else {
                    break;
                }
            }
            sample = colorSubscriber.recv_async() => {
                if let Ok(sample) = sample {
                    let color_frame: ColorFrameSerializable = bincode::decode_from_slice(sample.payload().to_bytes().as_ref(), bincode::config::standard()).unwrap().0;
                    println!("Received (color): {:?}", color_frame.data.len());
                } else {
                    break;
                }
            }
        }
    }
}