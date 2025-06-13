// src/rgbd_odom.rs
use nalgebra::{Isometry3, Matrix3, Translation3, Vector3, UnitQuaternion, Unit};
use kornia_icp::{icp_vanilla, ICPConvergenceCriteria, ICPResult};
use kornia_3d::pointcloud::PointCloud;
use rand::{seq::IteratorRandom, thread_rng};
use std::time::Instant;

use crate::types::{DepthFrameRealUnits, MotionFrameData};

/// Same struct you already used elsewhere.
pub use crate::types::Intrinsics;

/// Lightweight RGB‑D odometry helper.
pub struct RgbdOdometry {
    intr: Intrinsics,
    prev_cloud: Option<Vec<[f32; 3]>>,
    prev_ts: Option<f64>,
    world_T_cam: Isometry3<f32>,
}

impl RgbdOdometry {
    pub fn new(intr: Intrinsics) -> Self {
        Self {
            intr,
            prev_cloud: None,
            prev_ts: None,
            world_T_cam: Isometry3::identity(),
        }
    }

    /// Process a new RGB‑D frame and return the *current* pose of the camera
    /// in a fixed world coordinate frame.
    pub fn process(
        &mut self,
        depth: &DepthFrameRealUnits,
        gyro: Option<&MotionFrameData>,
    ) -> anyhow::Result<(Isometry3<f32>, Vec<[f32; 3]>)> {
        // 1. build point cloud ------------------------------------------------
        let mut cloud = depth_to_points(depth, &self.intr);

        // Randomly down‑sample to ~10 k points to keep ICP fast
        const MAX_POINTS: usize = 10_000;
        if cloud.len() > MAX_POINTS {
            let mut rng = thread_rng();
            cloud = cloud
                .into_iter()
                .choose_multiple(&mut rng, MAX_POINTS);
        }

        // 2. First frame?  Just cache and bail -------------------------------
        let Some(prev) = &self.prev_cloud else {
            self.prev_cloud = Some(cloud.clone());
            self.prev_ts = Some(depth.timestamp);
            return Ok((self.world_T_cam, cloud));
        };

        // 3. Build an initial guess from gyro Δθ -----------------------------
        let init_guess: nalgebra::Isometry<f32, Unit<nalgebra::Quaternion<f32>>, 3> = if let (Some(g), Some(last_ts)) = (gyro, self.prev_ts) {
            let dt = (depth.timestamp - last_ts) as f32;
            let omega = Vector3::from(g.gyro) * dt; // rad
            let angle = omega.norm();
            let axis = if angle > 1e-6 { 
                Unit::new_normalize(omega / angle)
            } else { 
                Unit::new_unchecked(Vector3::z())
            };
            Isometry3::identity()

        } else {
            Isometry3::identity()
        };

        // 4. Run ICP ---------------------------------------------------------
        let t0 = Instant::now();
        
        // Convert Vec<[f32; 3]> to Vec<[f64; 3]> for kornia-icp
        let prev_f64: Vec<[f64; 3]> = prev.iter().map(|&[x, y, z]| [x as f64, y as f64, z as f64]).collect();
        let cloud_f64: Vec<[f64; 3]> = cloud.iter().map(|&[x, y, z]| [x as f64, y as f64, z as f64]).collect();
        
        let src = PointCloud::new(prev_f64, None, None);
        let dst = PointCloud::new(cloud_f64, None, None);
        let criteria = ICPConvergenceCriteria { max_iterations: 20, tolerance: 1e-6 };
        
        // Convert initial guess to the format expected by icp_vanilla
        let init_rot_mat = init_guess.rotation.to_rotation_matrix();
        let init_rot_array: [[f64; 3]; 3] = [
            [init_rot_mat[(0, 0)] as f64, init_rot_mat[(0, 1)] as f64, init_rot_mat[(0, 2)] as f64],
            [init_rot_mat[(1, 0)] as f64, init_rot_mat[(1, 1)] as f64, init_rot_mat[(1, 2)] as f64],
            [init_rot_mat[(2, 0)] as f64, init_rot_mat[(2, 1)] as f64, init_rot_mat[(2, 2)] as f64],
        ];
        let init_trans_array: [f64; 3] = [
            init_guess.translation.x as f64,
            init_guess.translation.y as f64,
            init_guess.translation.z as f64,
        ];
        
        let ICPResult { rotation, translation, rmse, .. } = icp_vanilla(
            &src, 
            &dst, 
            init_rot_array,
            init_trans_array, 
            criteria
        ).map_err(|e| anyhow::anyhow!("ICP failed: {}", e))?;
        
        tracing::debug!("ICP in {:?}", t0.elapsed());

        // Check for degeneracy - if RMSE is too high or NaN, skip ICP update
        if rmse.is_nan() || rmse > 0.05 {
            tracing::warn!("ICP degenerate (rmse={rmse:.3}) – skipping, relying on IMU");
            self.prev_cloud = Some(cloud.clone());
            self.prev_ts = Some(depth.timestamp);
            return Ok((self.world_T_cam, cloud));
        }

        // 5. Accumulate pose --------------------------------------------------
        let rotation_f32: [[f32; 3]; 3] = [
            [rotation[0][0] as f32, rotation[0][1] as f32, rotation[0][2] as f32],
            [rotation[1][0] as f32, rotation[1][1] as f32, rotation[1][2] as f32],
            [rotation[2][0] as f32, rotation[2][1] as f32, rotation[2][2] as f32],
        ];
        let translation_f32: [f32; 3] = [
            translation[0] as f32,
            translation[1] as f32,
            translation[2] as f32,
        ];
        
        // kornia-icp outputs rotation in row-major; nalgebra expects column-major.
        let rot_matrix = Matrix3::from_column_slice(&[
            rotation_f32[0][0], rotation_f32[1][0], rotation_f32[2][0],
            rotation_f32[0][1], rotation_f32[1][1], rotation_f32[2][1],
            rotation_f32[0][2], rotation_f32[1][2], rotation_f32[2][2],
        ]);
        
        let delta = Isometry3::from_parts(
            Translation3::from(Vector3::from(translation_f32)),
            UnitQuaternion::from_matrix(&rot_matrix),
        );
        self.world_T_cam = self.world_T_cam * delta;

        // 6. Book‑keeping -----------------------------------------------------
        self.prev_cloud = Some(cloud.clone());
        self.prev_ts = Some(depth.timestamp);

        Ok((self.world_T_cam, cloud))
    }
}

/// Convert a 2‑D depth map into a Vec of 3‑D points in the camera frame.
fn depth_to_points(depth: &DepthFrameRealUnits, intr: &Intrinsics) -> Vec<[f32; 3]> {
    use rayon::prelude::*;
    let fx = intr.fx;
    let fy = intr.fy;
    let cx = intr.ppx;
    let cy = intr.ppy;
    (0..depth.height)
        .into_par_iter()
        .flat_map_iter(|v| {
            (0..depth.width).filter_map(move |u| {
                let d = depth.data[v * depth.width + u];
                if d.is_finite() && d >= 0.55 && d <= 8.0 {
                    let x = (u as f32 - cx) * d / fx;
                    let y = (v as f32 - cy) * d / fy;
                    Some([x, y, d])
                } else {
                    None
                }
            })
        })
        .collect()
}
