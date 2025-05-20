// odom.rs  (new task)
use opencv::{
    core::{KeyPoint, Mat, MatTraitConstManual, Vector},
    features2d,
    calib3d, types::VectorOfDMatch,
};
use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::prelude::MatTraitConst;
pub struct OdomState { pose_ws: Isometry3<f32> }

pub fn process_pair(rgb_prev: &Mat, rgb_curr: &Mat,
                    pts3d_prev: &[Vector3<f32>], intr: &Intrinsics,
                    state: &mut OdomState) -> Isometry3<f32> {
    // 1. ORB detect+describe
    let mut orb = features2d::ORB::create(
        1500, 1.2, 8, 31, 0, 2, features2d::ORB_HARRIS_SCORE, 31, 20
    ).unwrap();
    let (mut kp1, mut desc1) = (Vector::<KeyPoint>::new(), Mat::default());
    orb.detect_and_compute(rgb_prev, &Mat::default(), &mut kp1, &mut desc1, false).unwrap();
    let (mut kp2, mut desc2) = (Vector::<KeyPoint>::new(), Mat::default());
    orb.detect_and_compute(rgb_curr, &Mat::default(), &mut kp2, &mut desc2, false).unwrap();

    // 2. match & Lowe ratio
    let mut matcher = features2d::BFMatcher::create(features2d::NORM_HAMMING, false).unwrap();
    let mut knn = types::VectorOfVectorOfDMatch::new();
    matcher.knn_train_match(&desc1, &desc2, &mut knn, 2, &Mat::default(), false).unwrap();
    let good: Vec<_> = knn.into_iter()
        .filter_map(|m| { if m.len()==2 && m[0].distance < 0.75*m[1].distance { Some(m[0]) } else { None }})
        .collect();

    // 3. build 3D–2D correspondences
    let mut obj_pts = Vec::<_>::new();
    let mut img_pts = Vec::<_>::new();
    for dm in &good {
        let p3 = pts3d_prev[dm.query_idx as usize];
        if p3.z > 0.0 {
            obj_pts.push(opencv::core::Point3f::new(p3.x, p3.y, p3.z));
            let kp = kp2.get(dm.train_idx).unwrap();
            img_pts.push(opencv::core::Point2f::new(kp.pt.x, kp.pt.y));
        }
    }

    // 4. PnP + RANSAC
    let cam_mat = opencv::core::Mat::from_slice_2d(&[
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ]).unwrap();
    let mut rvec = Mat::default(); let mut tvec = Mat::default(); let mut inliers = Mat::default();
    calib3d::solve_pnp_ransac(
        &obj_pts, &img_pts, &cam_mat, &Mat::default(),
        &mut rvec, &mut tvec, false, 100, 3.0, 0.99, &mut inliers, calib3d::SOLVEPNP_ITERATIVE
    ).unwrap();

    // 5. convert to nalgebra
    let rmat = {
        let mut m = Mat::default();
        calib3d::rodrigues(&rvec, &mut m, &mut Mat::default()).unwrap();
        Matrix3::from_iterator(m.data_typed::<f64>().unwrap().iter().map(|v| *v as f32))
    };
    let t = Vector3::new(*tvec.at_2d::<f64>(0,0).unwrap() as f32,
                         *tvec.at_2d::<f64>(1,0).unwrap() as f32,
                         *tvec.at_2d::<f64>(2,0).unwrap() as f32);
    let xi = Isometry3::from_parts(t.into(), rmat.into());

    state.pose_ws *= xi;   // accumulate world-scale pose
    state.pose_ws
}
