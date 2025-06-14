// This file can be used for testing the library functionality from Rust
// The main functionality is now exposed through lib.rs as a Python module

mod types;

use types::*;

fn main() {
    println!("Zenoh D435i Subscriber Library");
    println!("This library is designed to be used from Python via PyO3.");
    println!("Example usage:");
    println!("  import zenoh_d435i_subscriber");
    println!("  subscriber = zenoh_d435i_subscriber.ZenohD435iSubscriber()");
    println!("  subscriber.connect()");
    println!("  subscriber.start_subscribing()");
}


