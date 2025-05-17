use nalgebra::Vector3;

use crate::types::{DepthFrameRealUnits, Extrinsics, Intrinsics};

/// Returns a depth map in RGB resolution.  Missing pixels are `f32::NAN`.
pub fn align_depth_to_color(
    depth: &DepthFrameRealUnits,
    d_intr: &Intrinsics,
    c_intr: &Intrinsics,
    extr: &Extrinsics,
) -> DepthFrameRealUnits {
    let mut out = vec![f32::NAN; c_intr.width * c_intr.height];

    for v in 0..depth.height {
        for u in 0..depth.width {
            let d = depth.data[v * depth.width + u];
            if !d.is_finite() || d <= 0.0 { continue; }

            // 1. de-project
            let x = (u as f32 - d_intr.ppx) * d / d_intr.fx;
            let y = (v as f32 - d_intr.ppy) * d / d_intr.fy;
            let z = d;

            // 2. transform
            let p_d = Vector3::new(x, y, z);
            let p_c = extr.rotation * p_d + extr.translation;

            // 3. project
            let u_c = ((p_c.x / p_c.z) * c_intr.fx + c_intr.ppx).round() as isize;
            let v_c = ((p_c.y / p_c.z) * c_intr.fy + c_intr.ppy).round() as isize;

            if u_c >= 0 && u_c < c_intr.width as isize &&
               v_c >= 0 && v_c < c_intr.height as isize {
                let idx = v_c as usize * c_intr.width + u_c as usize;
                // keep nearest sample to handle occlusion
                if !out[idx].is_finite() || p_c.z < out[idx] {
                    out[idx] = p_c.z;
                }
            }
        }
    }
    DepthFrameRealUnits {
        width: c_intr.width,
        height: c_intr.height,
        data: out,
        timestamp: depth.timestamp,
    }
}
