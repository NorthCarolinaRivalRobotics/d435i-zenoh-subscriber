use crate::types::{DepthFrameRealUnits, MotionFrameData};

pub struct StampedTriple {
    pub depth:  DepthFrameRealUnits,
    pub colour: (Vec<u8>, f64),   // jpeg + ts
    pub motion: MotionFrameData,
}

// simple channel consumer
pub async fn run(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<StampedTriple>
) {
    while let Some(bundle) = rx.recv().await {
        // TODO: call your real odometry code here
        println!("Got synchronised triple @ {:.3}", bundle.depth.timestamp);
    }
}
