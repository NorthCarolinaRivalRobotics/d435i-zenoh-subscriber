use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about)]
pub struct Args {
    /// Record an on-disk log while running live
    #[arg(long)]
    pub start_log: Option<PathBuf>,

    /// Play back a previously recorded log instead of connecting to Zenoh
    #[arg(long)]
    pub play_log:  Option<PathBuf>,
}

impl Args {
    pub fn mode(&self) -> RunMode {
        match (&self.start_log, &self.play_log) {
            (Some(_), Some(_)) =>
                panic!("--start-log and --play-log are mutually exclusive"),
            (Some(p), None) => RunMode::LiveWithLog(p.clone()),
            (None,  Some(p)) => RunMode::Playback(p.clone()),
            (None,  None)    => RunMode::LiveNoLog,
        }
    }
}

#[derive(Clone)]
pub enum RunMode {
    LiveNoLog,
    LiveWithLog(PathBuf),
    Playback(PathBuf),
}
