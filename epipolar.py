import cv2
import numpy as np
import json
import os
import argparse
import viser
import viser.transforms as vtf

def get_green_dots(image_path, min_area=5):
    """
    Detects green dots in the given image.
    Accounts for anti-aliasing by thresholding in HSV space.
    Returns a list of (x, y) coordinates of the centroids.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green color
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        # Filter out very small noise
        if cv2.contourArea(contour) >= min_area:
            # Calculate moments for each contour to find the centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                points.append((cX, cY))
    
    return np.array(points)

def get_intrinsics(resolution, fov_deg):
    """
    Compute intrinsic matrix K from resolution and field of view.
    Assuming square pixels and fov_deg is the full field of view (horizontal/vertical since it's square).
    """
    w, h = resolution
    fov_rad = np.deg2rad(fov_deg)
    focal_length = (w / 2.0) / np.tan(fov_rad / 2.0)
    
    cx = w / 2.0
    cy = h / 2.0
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K

def compute_fundamental_matrix(K, w2c_1, c2w_1, w2c_2, c2w_2, is_opengl=True):
    """
    Computes the Fundamental Matrix F that maps points from view 1 to epipolar lines in view 2.
    """
    # Relative pose from 1 to 2
    # X_c2 = w2c_2 * c2w_1 * X_c1
    w2c_2_mat = np.array(w2c_2)
    c2w_1_mat = np.array(c2w_1)
    
    # Transformation to swap coordinate conventions if necessary
    # OpenGL/Blender is +Y up, -Z forward. OpenCV is +Y down, +Z forward.
    T = np.eye(4)
    if is_opengl:
        T[1, 1] = -1
        T[2, 2] = -1
        
    M = T @ w2c_2_mat @ c2w_1_mat @ np.linalg.inv(T)
    
    R = M[:3, :3]
    t = M[:3, 3]
    
    # Skew-symmetric matrix of t
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    
    # Essential Matrix
    E = tx @ R
    
    # Fundamental Matrix: F = K^-T * E * K^-1
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    
    return F, M

def draw_epipolar_lines(img, lines, pts, colors):
    """
    Draws epipolar lines and points on the image.
    img: image on which to draw
    lines: epipolar lines in format (a, b, c) where ax + by + c = 0
    pts: points corresponding to the lines (optional, for overlaying the actual detections)
    colors: list of colors for each line/point
    """
    h, w = img.shape[:2]
    out_img = img.copy()
    
    for r_line, color in zip(lines, colors):
        color = tuple(map(int, color))
        if abs(r_line[1]) > abs(r_line[0]):
            # Line is more horizontal
            x0, x1 = 0, w
            y0 = -r_line[2] / r_line[1]
            y1 = -(r_line[2] + r_line[0] * w) / r_line[1]
        else:
            # Line is more vertical
            y0, y1 = 0, h
            x0 = -r_line[2] / r_line[0]
            x1 = -(r_line[2] + r_line[1] * h) / r_line[0]
            
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        cv2.line(out_img, (x0, y0), (x1, y1), color, 1)
        
    for pt in pts:
        cv2.circle(out_img, tuple(map(int, pt)), 5, (0, 0, 255), -1) # Draw detected points in red
        
    return out_img

def draw_colored_points(img, pts, colors):
    """
    Draws points on the image with specified colors.
    """
    out_img = img.copy()
    for pt, color in zip(pts, colors):
        color = tuple(map(int, color))
        cv2.circle(out_img, tuple(map(int, pt)), 5, color, -1)
    return out_img

def recover_camera_pose(pts1_matched, pts2_matched, K):
    """
    Computes Essential Matrix using RANSAC and recovers relative R, t.
    Returns R_est, t_est, and the mask of inliers.
    """
    if len(pts1_matched) < 5:
        print("Not enough matches to compute Essential Matrix.")
        return None, None, None
        
    # Use RANSAC to compute Essential Matrix
    E, mask = cv2.findEssentialMat(
        pts1_matched, pts2_matched, K,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    
    if E is None:
        print("Failed to compute Essential Matrix.")
        return None, None, None
        
    # Recover Pose (R, t) from Essential Matrix
    # Using only inliers from RANSAC
    _, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1_matched, pts2_matched, K, mask=mask)
    
    return R_est, t_est, mask

def triangulate_points(pts1, pts2, K, R, t):
    """
    Triangulates 3D points from 2D correspondences and relative camera pose.
    Camera 1 is assumed to be at the origin [I | 0].
    Camera 2 is at [R | t].
    """
    # Projection matrix for Camera 1
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # Projection matrix for Camera 2
    P2 = K @ np.hstack((R, t))
    
    # Triangulate points (returns homogeneous 4D coordinates)
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous to Cartesian (divide by W)
    pts3d = pts4d[:3, :] / pts4d[3, :]
    
    return pts3d.T

def normalize_points(pts3d):
    """
    Normalizes a set of 3D points to fit within a unit cube centered at origin.
    Returns the normalized points, the centroid, and the scale factor used.
    """
    centroid = np.mean(pts3d, axis=0)
    centered_pts = pts3d - centroid
    
    # Find max absolute distance from origin to fit in [-1, 1] cube
    max_dist = np.max(np.abs(centered_pts))
    
    if max_dist > 0:
        scale = 1.0 / max_dist
    else:
        scale = 1.0
        
    normalized_pts = centered_pts * scale
    
    return normalized_pts, centroid, scale

def visualize_3d(pts3d, colors, R, t, K, img1, img2, resolution, mask=None):
    """
    Visualizes the 3D points and cameras using Viser.
    """
    server = viser.ViserServer()
    
    # Filter out RANSAC outliers
    if mask is not None:
        valid_indices = mask.ravel() == 1
        pts3d = pts3d[valid_indices]
        colors = [colors[i] for i in range(len(colors)) if valid_indices[i]]
        
    # Convert BGR BGR to RGB 0-255 for Viser
    rgb_colors = []
    for c in colors:
        rgb_colors.append([c[2], c[1], c[0]])
    rgb_colors = np.array(rgb_colors, dtype=np.uint8)
    
    # Normalize the 3D points to fit in a unit cube
    norm_pts, centroid, scale = normalize_points(pts3d)
    
    # Register the point cloud
    server.scene.add_point_cloud(
        "skeleton_points", 
        norm_pts, 
        colors=rgb_colors, 
        point_size=0.02
    )
    
    # Calculate Camera Parameters for Viser
    w, h = resolution
    fx = K[0, 0]
    fy = K[1, 1]
    
    # Convert focal length to field of view for Viser
    # FOV in radians = 2 * arctan(width / (2 * focal_length))
    fov_y = 2.0 * np.arctan(h / (2.0 * fy))
    aspect = w / h
    
    # Convert BGR to RGB for Viser image overlay
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # ----- Camera 1 (Source) -----
    # Located at scaled/translated origin
    cam1_pos = -centroid * scale
    cam1_quat = (1.0, 0.0, 0.0, 0.0) # Identity rotation (wxyz)
    
    server.scene.add_camera_frustum(
        "cameras/cam1",
        fov=fov_y,
        aspect=aspect,
        scale=0.5,
        image=img1_rgb,
        position=cam1_pos,
        wxyz=cam1_quat,
        color=(255, 0, 0) # Red
    )
    
    # ----- Camera 2 (Target) -----
    # Original pose from R, t:  X_c = R * X_w + t
    # Therefore: X_w = R^T * X_c - R^T * t
    # Camera center in world coords is -R^T * t
    cam2_center_w = -R.T @ t
    
    # Normalize camera 2 position using the same scale/centroid
    cam2_pos = (cam2_center_w.flatten() - centroid) * scale
    
    # The orientation of the camera in world coordinates is R^T
    cam2_quat = vtf.SO3.from_matrix(R.T).wxyz
    
    server.scene.add_camera_frustum(
        "cameras/cam2",
        fov=fov_y,
        aspect=aspect,
        scale=0.5,
        image=img2_rgb,
        position=cam2_pos,
        wxyz=cam2_quat,
        color=(0, 255, 0) # Green
    )
    
    print("\n--- Viser 3D Viewer ---")
    print("Viser server running at http://localhost:8080")
    print("Press Ctrl+C in this terminal to exit.")
    
    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Closing Viser viewer...")

def draw_correspondences(img1, img2, pts1, pts2, colors, mask=None):
    """
    Draws points from both images side-by-side and connects matched points with lines.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side image
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2
    
    for i in range(len(pts1)):
        pt1 = tuple(map(int, pts1[i]))
        pt2 = tuple(map(int, pts2[i]))
        color_pt = tuple(map(int, colors[i]))
        
        # Offset x coordinate for second image
        pt2_offset = (pt2[0] + w1, pt2[1])
        
        # Draw points
        cv2.circle(out_img, pt1, 5, color_pt, -1) # Original color for image 1
        cv2.circle(out_img, pt2_offset, 5, (0, 255, 0), -1) # Green for image 2
        
        # Draw connecting line
        line_color = (128, 128, 128) # Grey default
        thickness = 1
        
        # Highlight inliers if mask is provided
        if mask is not None:
            if mask[i]:
                line_color = (200, 200, 200) # Light grey for RANSAC inliers
                thickness = 1
            else:
                line_color = (64, 64, 64) # Dark grey for outliers
                
        cv2.line(out_img, pt1, pt2_offset, line_color, thickness)
        
    return out_img

def main():
    parser = argparse.ArgumentParser(description="Visualize epipolar lines from 3D projected keypoints")
    parser.add_argument("--json", type=str, default="data/cameras.json", help="Path to cameras.json")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first image (source of points)")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second image (target for epipolar lines)")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory to save visualizations")
    parser.add_argument("--opengl", action=argparse.BooleanOptionalAction, default=True, help="Use OpenGL/Blender coordinate convention (+Y up, -Z forward)")
    parser.add_argument("--opencv", action="store_false", dest="opengl", help="Use OpenCV coordinate convention (+Y down, +Z forward)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Distance threshold (pixels) for epipolar correspondence")
    parser.add_argument("--no-3d", action="store_true", help="Disable the 3D Polyscope interactive visualization")
    parser.add_argument("--use-gt-pose", action="store_true", help="Use ground truth pose from JSON instead of estimated RANSAC pose for 3D triangulation")

    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    with open(args.json, 'r') as f:
        cam_data = json.load(f)
        
    frames = {frame['view_id']: frame for frame in cam_data['frames']}
    
    # Extract view IDs from filenames (e.g., 'data/skels/00.png' -> 0)
    view1_id = int(os.path.splitext(os.path.basename(args.img1))[0])
    view2_id = int(os.path.splitext(os.path.basename(args.img2))[0])
    
    frame1 = frames[view1_id]
    frame2 = frames[view2_id]
    
    # 1. Get 2D points from Image 1
    pts1 = get_green_dots(args.img1)
    
    # Also get 2D points from Image 2 just for visualization overlay
    pts2 = get_green_dots(args.img2)
    
    if len(pts1) == 0:
        print(f"No points detected in {args.img1}")
        return
        
    # 2. Camera matrices
    # Compute K (assumed same for both views)
    K = get_intrinsics(frame1['resolution'], frame1['fov_deg'])
    
    # Compute Fundamental Matrix F
    F, M = compute_fundamental_matrix(
        K, 
        frame1['w2c'], frame1['c2w'],
        frame2['w2c'], frame2['c2w'],
        is_opengl=args.opengl
    )
    
    # 3. Compute Epipolar Lines
    # For a point x1 in img1, the epipolar line l2 in img2 is F @ x1
    # Convert points to homogeneous coordinates
    pts1_h = np.hstack((pts1, np.ones((len(pts1), 1))))
    
    # Calculate lines
    lines2 = (F @ pts1_h.T).T
    
    # Generate random colors for the lines/points
    # We use HSV to generate visually distinct colors
    colors = []
    for i in range(len(pts1)):
        hue = int((i / len(pts1)) * 179) # OpenCV hue is 0-179
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        color_tuple = color_bgr[0, 0].tolist()
        colors.append(color_tuple)
    
    # 4. Draw & Save
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    
    img1_name = os.path.splitext(os.path.basename(args.img1))[0]
    img2_name = os.path.splitext(os.path.basename(args.img2))[0]
    
    out_img1 = draw_colored_points(img1, pts1, colors)
    out_img2 = draw_epipolar_lines(img2, lines2, pts2, colors)
    
    # 5. Evaluate distances for debugging and Establish Correspondences
    matched_pts1 = []
    matched_pts2 = []
    matched_colors = []
    
    if len(pts2) > 0:
        pts2_h = np.hstack((pts2, np.ones((len(pts2), 1))))
        distances = []
        for i, l in enumerate(lines2):
            # Normalize line
            l_norm = l / np.sqrt(l[0]**2 + l[1]**2)
            # Find distances from all pts2 to this epipolar line
            dists = np.abs(pts2_h @ l_norm)
            
            # Find the minimum distance
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            distances.append(min_dist)
            
            # If distance is within threshold, consider it a candidate correspondence
            if min_dist <= args.threshold:
                matched_pts1.append(pts1[i])
                matched_pts2.append(pts2[min_idx])
                matched_colors.append(colors[i])
                
        print(f"\n--- Correspondence Analysis ---")
        print(f"Mean distance from epipolar lines to closest points in image 2: {np.mean(distances):.2f} pixels")
        print(f"Found {len(matched_pts1)} candidate correspondences within threshold {args.threshold}px")
        
        # 6. Run RANSAC to compute Camera Transformation
        matched_pts1 = np.array(matched_pts1)
        matched_pts2 = np.array(matched_pts2)
        
        if len(matched_pts1) >= 5:
            print("\n--- RANSAC Camera Pose Estimation ---")
            R_est, t_est, mask = recover_camera_pose(matched_pts1, matched_pts2, K)
            
            if R_est is not None and t_est is not None:
                print("\nEstimated R:")
                print(R_est)
                print("\nEstimated t (normalized):")
                print(t_est)
                
                # Visualize Correspondences
                out_matches_img = draw_correspondences(img1, img2, matched_pts1, matched_pts2, matched_colors, mask=mask)
                out_matches_name = f"matches_{img1_name}_to_{img2_name}.png"
                out_matches_path = os.path.join(args.out_dir, out_matches_name)
                cv2.imwrite(out_matches_path, out_matches_img)
                print(f"Saved correspondence visualization to {out_matches_path}")
                
                # Compare with Ground Truth M (from step 2)
                # Note: M is the full 4x4 matrix mapping from View 1 to View 2
                R_gt = M[:3, :3]
                t_gt = M[:3, 3].reshape(3, 1)
                
                # t_gt needs to be normalized to compare with t_est
                t_gt_norm = t_gt / np.linalg.norm(t_gt)
                
                print("\nGround Truth R:")
                print(R_gt)
                print("\nGround Truth t (normalized):")
                print(t_gt_norm)
                
                # Compute error
                # R error: angle of R_est * R_gt^T
                R_err_mat = R_est @ R_gt.T
                # Trace(R) = 1 + 2*cos(theta)
                trace = np.trace(R_err_mat)
                angle_err = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))
                print(f"\nRotation Error (degrees): {np.rad2deg(angle_err):.4f}")
                
                # t error: angle between normalized vectors
                cos_t_err = np.sum(t_est * t_gt_norm)
                t_angle_err = np.arccos(np.clip(cos_t_err, -1.0, 1.0))
                print(f"Translation Direction Error (degrees): {np.rad2deg(t_angle_err):.4f}")
                
                # 7. 3D Triangulation and Visualization
                if not args.no_3d:
                    if args.use_gt_pose:
                        print("\nTriangulating using Ground Truth Pose...")
                        pts3d = triangulate_points(matched_pts1, matched_pts2, K, R_gt, t_gt)
                        visualize_3d(pts3d, matched_colors, R_gt, t_gt, K, img1, img2, frame1['resolution'], mask=mask)
                    else:
                        print("\nTriangulating using Estimated RANSAC Pose...")
                        pts3d = triangulate_points(matched_pts1, matched_pts2, K, R_est, t_est)
                        visualize_3d(pts3d, matched_colors, R_est, t_est, K, img1, img2, frame1['resolution'], mask=mask)
        else:
            print("Not enough candidate correspondences for RANSAC (< 5).")
    
    # Save Image 1 with colored points
    out_pts_name = f"pts_{img1_name}.png"
    out_pts_path = os.path.join(args.out_dir, out_pts_name)
    cv2.imwrite(out_pts_path, out_img1)
    print(f"Saved points visualization to {out_pts_path}")
    
    # Save Image 2 with epipolar lines
    out_epi_name = f"epi_{img1_name}_to_{img2_name}_opengl_{args.opengl}.png"
    out_epi_path = os.path.join(args.out_dir, out_epi_name)
    cv2.imwrite(out_epi_path, out_img2)
    print(f"Saved epipolar lines visualization to {out_epi_path}")

if __name__ == "__main__":
    main()
