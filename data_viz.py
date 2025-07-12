
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def plot_losses(train_losses, title="Training Loss", ylabel="Loss"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')


def visualize_dataset(rgb_path, mask_path, bbox3d_path, pc_path):
    # Load data
    rgb = plt.imread(rgb_path)

    mask = np.load(mask_path)
    mask = np.any(mask, axis=0, keepdims=True).squeeze()

    bbox3d = np.load(bbox3d_path)  
    point_cloud = np.load(pc_path)
    point_cloud = point_cloud.reshape(3, -1).T  # Transpose to get (N, 3)
    
    # Visualize RGB image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Mask Image")
    plt.axis("off")

    plt.show()

    # Visualize 3D point cloud with bounding box
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(point_cloud)

    # Create bounding boxes
    geometries = [pc_o3d]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
    ]
    for i in range(bbox3d.shape[0]): 
        single_bbox = bbox3d[i]  
        if single_bbox.dtype != np.float32 and single_bbox.dtype != np.float64:
            single_bbox = single_bbox.astype(np.float32)

        # Create LineSet for the bounding box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(single_bbox)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 0, 0]) 
        geometries.append(line_set)

    # Visualize all geometries
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud + 3D Bounding Boxes")


if __name__=="__main__":
    file_name = "8b061a90-9915-11ee-9103-bbb8eae05561"
    rgb_path = f"./data/{file_name}/rgb.jpg"
    mask_path = f"./data/{file_name}/mask.npy"
    bbox3d_path = f"./data/{file_name}/bbox3d.npy"
    pc_path = f"./data/{file_name}/pc.npy"
    visualize_dataset(rgb_path, mask_path, bbox3d_path, pc_path)
