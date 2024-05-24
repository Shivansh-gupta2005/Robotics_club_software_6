import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

# Load RGB and Depth images
def load_images(rgb_path, depth_path):
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    return rgb_image, depth_image

# Generate point cloud from depth image
def generate_point_cloud(rgb_image, depth_image, fx, fy, cx, cy):
    points = []
    colors = []

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            Z = depth_image[v, u] / 1000.0  # Convert depth to meters
            if np.allclose(Z, 0): 
                continue  # Skip invalid depth values
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(rgb_image[v, u])

    if not points:  # Check if points list is empty
        print("No valid points found. Returning None.")
        return None

    points = np.array(points)
    colors = np.array(colors) / 255.0  # Normalize colors to [0, 1]

    print("Points shape:", points.shape)
    print("Points dtype:", points.dtype)

    # Reshape points array to remove extra dimension
    points = points.reshape(-1, 3)

    try:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    except Exception as e:
        print("Error:", e)

    return point_cloud


# Custom ground detection algorithm
def detect_ground(point_cloud, height_threshold=0.1):
    points = np.asarray(point_cloud.points)
    ground_indices = points[:, 1] < height_threshold

    ground_cloud = point_cloud.select_by_index(np.where(ground_indices)[0])
    non_ground_cloud = point_cloud.select_by_index(np.where(~ground_indices)[0])

    return ground_cloud, non_ground_cloud
# Visualize point clouds
def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])

# Main pipeline
def main(rgb_path, depth_path, fx, fy, cx, cy):
    rgb_image, depth_image = load_images(rgb_path, depth_path)
    point_cloud = generate_point_cloud(rgb_image, depth_image, fx, fy, cx, cy)

    if point_cloud is None:
        print("Failed to generate point cloud. Exiting.")
        return

    visualize_point_cloud(point_cloud)

    # Wait for visualization window to close
    o3d.visualization.webrtc_server.enable_webrtc()

    input("Press Enter to close the visualization...")

# Example usage
# Calibration parameters (example values, adjust based on your camera)
fx, fy = 525.0, 525.0  # Focal length
cx, cy = 319.5, 239.5  # Principal point

# Paths to your RGB and depth images
rgb_path = 'rgb.jpg'
depth_path = 'depth.jpg'

main(rgb_path, depth_path, fx, fy, cx, cy)


