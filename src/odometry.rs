use crate::rgbd_odom::{Intrinsics, RgbdOdometry};
use crate::types::{DepthFrameRealUnits, MotionFrameData};
use rerun::archetypes::{Transform3D, Points3D, Pinhole};
use nalgebra::{Point3, Isometry3};

#[derive(Clone)]
pub struct StampedTriple {
    pub depth: DepthFrameRealUnits,
    pub colour: (Vec<u8>, f64),        // jpeg + ts
    pub motion: MotionFrameData,
}

/// Extract RGB color for a point based on its camera coordinates
fn get_color_for_point(
    world_pos: [f32; 3], 
    rgb_data: &[u8], 
    rgb_width: usize, 
    rgb_height: usize,
    intrinsics: &Intrinsics,
    world_T_cam: &nalgebra::Isometry3<f32>
) -> Option<[u8; 3]> {
    // Transform from world coordinates back to camera coordinates
    let cam_T_world = world_T_cam.inverse();
    let world_point = nalgebra::Point3::new(world_pos[0], world_pos[1], world_pos[2]);
    let cam_point = cam_T_world * world_point;
    
    // Project to pixel coordinates
    if cam_point.z <= 0.0 { return None; }
    
    let pixel_x = ((cam_point.x / cam_point.z) * intrinsics.fx + intrinsics.ppx).round() as i32;
    let pixel_y = ((cam_point.y / cam_point.z) * intrinsics.fy + intrinsics.ppy).round() as i32;
    
    // Check if pixel is within image bounds
    if pixel_x < 0 || pixel_x >= rgb_width as i32 || pixel_y < 0 || pixel_y >= rgb_height as i32 {
        return None;
    }
    
    // Get RGB values from the image data (RGB format: 3 bytes per pixel)
    let pixel_idx = (pixel_y as usize * rgb_width + pixel_x as usize) * 3;
    if pixel_idx + 2 < rgb_data.len() {
        Some([
            rgb_data[pixel_idx],     // R
            rgb_data[pixel_idx + 1], // G
            rgb_data[pixel_idx + 2], // B
        ])
    } else {
        None
    }
}

pub async fn run(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<StampedTriple>,
    rec: rerun::RecordingStream,
) {
    // 1️⃣  create odometry helper once
    let cam = Intrinsics {
        width: 640,
        height: 480,
        fx: 607.2676,
        fy: 607.149,
        ppx: 316.65408,
        ppy: 244.13338,
    };
    let mut odo = RgbdOdometry::new(cam);

    // Log a reference point at the origin to show where we started
    rec.log(
        "world/origin",
        &Points3D::new([[0.0, 0.0, 0.0]])
            .with_colors([rerun::Color::from_rgb(255, 0, 0)]) // Red point
            .with_radii([0.01]), // Much smaller radius
    ).unwrap();

    // 2️⃣  consume the channel
    while let Some(bundle) = rx.recv().await {
        if let Ok((world_T_cam, cloud)) = odo.process(&bundle.depth, Some(&bundle.motion)) {
            // translation & 3×3 rotation matrix → Transform3D
            let t = world_T_cam.translation.vector;
            let r = world_T_cam.rotation.to_rotation_matrix();
            let mat3x3: [[f32; 3]; 3] = [
                [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
                [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
                [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
            ];
            
            // Log the camera transform
            rec.log(
                "world/camera",
                &Transform3D::from_translation_mat3x3(
                    [t.x, t.y, t.z],
                    mat3x3,
                ),
            ).unwrap();

            // Log pinhole camera frustum to a separate entity that inherits the transform
            rec.log(
                "world/camera/frustum",
                &Pinhole::from_focal_length_and_resolution(
                    [cam.fx, cam.fy],
                    [cam.width as f32, cam.height as f32],
                )
                .with_principal_point([cam.ppx, cam.ppy])
                .with_camera_xyz(rerun::components::ViewCoordinates::RDF), // Right-Down-Forward
            ).unwrap();

            // Decompress the JPEG RGB image
            if let Ok(rgb_img) = turbojpeg::decompress_image::<turbojpeg::image::Rgb<u8>>(&bundle.colour.0) {
                let rgb_data = rgb_img.into_raw();
                let (rgb_width, rgb_height) = (cam.width, cam.height); // Assuming RGB and depth have same resolution
                
                // Extract colors for each point in the cloud
                let colors: Vec<rerun::Color> = cloud.iter().map(|&point| {
                    if let Some(rgb) = get_color_for_point(point, &rgb_data, rgb_width, rgb_height, &cam, &world_T_cam) {
                        rerun::Color::from_rgb(rgb[0], rgb[1], rgb[2])
                    } else {
                        // Default gray color for points without valid RGB mapping
                        rerun::Color::from_rgb(128, 128, 128)
                    }
                }).collect();

                // point cloud with colors, inherits the transform
                rec.log(
                    "world/camera/points", 
                    &Points3D::new(cloud).with_colors(colors)
                ).unwrap();
            } else {
                // Fallback: log point cloud without colors if JPEG decompression fails
                rec.log("world/camera/points", &Points3D::new(cloud)).unwrap();
            }
        }
    }
}
