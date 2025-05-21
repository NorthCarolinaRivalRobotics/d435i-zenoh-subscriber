use std::collections::BTreeMap;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use ordered_float::OrderedFloat;

use crate::types::{Frame};
use crate::odometry::StampedTriple;

const TOL: f64 = 0.1;      // 50 ms tolerance. tune later.
type Stamp = OrderedFloat<f64>;      // one alias, readable later

pub async fn run(
    mut rx: UnboundedReceiver<Frame>,
    odom_tx: UnboundedSender<StampedTriple>,
) {
    use std::collections::BTreeMap;

    let (mut depth_q, mut color_q, mut motion_q):
        (BTreeMap<Stamp, Frame>, BTreeMap<Stamp, Frame>, BTreeMap<Stamp, Frame>) =
        Default::default();

    while let Some(frm) = rx.recv().await {
        let ts_sec = frm.ts() / 1_000.0;
        let ts     = Stamp::from(ts_sec);
    
        match &frm {
            Frame::Depth(_)  => { depth_q.insert(ts, frm);  }
            Frame::Color(_)  => { color_q.insert(ts, frm);  }
            Frame::Motion(_) => { motion_q.insert(ts, frm); }
        };

        if let Some((&t, _)) = depth_q.iter().next() {
            let colour = nearest(&color_q, t);
            let motion = nearest(&motion_q, t);
            if depth_q.len() % 60 == 0 {          // once per ~2 s at 30 FPS
                let t = *depth_q.keys().next().unwrap();
                let dc  = nearest(&color_q,  t);
                let dm  = nearest(&motion_q, t);
                println!(
                    "earliest D={:.6}  nearest C={:?}  M={:?}   Δdc={:.3} ms  Δdm={:.3} ms",
                    t.into_inner(),
                    dc.map(|x| x.into_inner()),
                    dm.map(|x| x.into_inner()),
                    dc.map_or(-1.0, |x| (x.into_inner()-t.into_inner())*1e3),
                    dm.map_or(-1.0, |x| (x.into_inner()-t.into_inner())*1e3),
                );
            }
            
            if let (Some(tc), Some(tm)) = (colour, motion) {
                if (t.into_inner() - tc.into_inner()).abs() < TOL
                    && (t.into_inner() - tm.into_inner()).abs() < TOL
                {
                    let depth  = match depth_q.remove(&t).unwrap()  { Frame::Depth(d) => d, _ => unreachable!() };
                    let colour = match color_q.remove(&tc).unwrap() { Frame::Color(c) => c, _ => unreachable!() };
                    let motion = match motion_q.remove(&tm).unwrap(){ Frame::Motion(m)=> m, _ => unreachable!() };
                    let triple: StampedTriple = StampedTriple { depth, colour, motion };
                    let _ = odom_tx.send(triple);
                }
            }
        }
    }
}

// O(log n) nearest neighbour (unchanged except for Stamp)
fn nearest(map: &BTreeMap<Stamp, Frame>, t: Stamp) -> Option<Stamp> {
    map.range(..=t).last()
       .into_iter()
       .chain(map.range(t..).next())
       .min_by(|a, b| (a.0.into_inner() - t.into_inner()).abs()
                     .partial_cmp(&(b.0.into_inner() - t.into_inner()).abs()).unwrap())
       .map(|(k, _)| *k)
}
