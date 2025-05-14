use types::DepthFrameSerializable;

mod types;

#[tokio::main]
async fn main() {
    let session = zenoh::open(zenoh::Config::default()).await.unwrap();
    let subscriber = session.declare_subscriber("camera/depth").await.unwrap();
    while let Ok(sample) = subscriber.recv_async().await {
        let depth_frame: DepthFrameSerializable = bincode::decode_from_slice(sample.payload().to_bytes().as_ref(), bincode::config::standard()).unwrap().0;
        println!("Received: {:?}", depth_frame.data.len());
    };
}