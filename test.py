
import cv2
import torch
import numpy as np
import open3d as o3d
import albumentations as A
from data_prepare import preprocess_image, pc_feature_extractor

if __name__=="__main__":
    model = torch.load('model.pt')
    file_name = "8b061a8b-9915-11ee-9103-bbb8eae05561"
    rgb_path = f"./data/{file_name}/rgb.jpg"
    mask_path = f"./data/{file_name}/mask.npy"
    bbox3d_path = f"./data/{file_name}/bbox3d.npy"
    pc_path = f"./data/{file_name}/pc.npy"

    rgb = cv2.imread(rgb_path)
    o_h, o_w = rgb.shape[0], rgb.shape[1]
    rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
    rgb = preprocess_image(rgb)
    rgb = rgb.unsqueeze(0)

    mask = np.load(mask_path)
    mask = np.any(mask, axis=0, keepdims=True).squeeze()
    mask_uint8 = mask.astype(np.uint8)
    mask = A.Resize(480, 640)(image=mask_uint8)["image"]
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)

    pc = np.load(pc_path).reshape(3, -1).T
    pointcloud_reshaped = pc.reshape(o_h, o_w, 3)
    resized_x = A.Resize(480, 640)(image=pointcloud_reshaped[:, :, 0])["image"]
    resized_y = A.Resize(480, 640)(image=pointcloud_reshaped[:, :, 1])["image"]
    resized_z = A.Resize(480, 640)(image=pointcloud_reshaped[:, :, 2])["image"]
    pc = np.stack([resized_x, resized_y, resized_z], axis=-1).reshape(-1, 3)
    pc_features = pc_feature_extractor(pc)
    pc_features = np.expand_dims(pc_features, 0)
    pc_features = torch.from_numpy(pc_features)

    pred_box = model([rgb, mask, pc_features])
    pred_box = pred_box[0].detach().numpy()

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    geometries = [pc_o3d]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
    ]
    for i in range(pred_box.shape[0]): 
        single_bbox = pred_box[i]  
        if single_bbox.dtype != np.float32 and single_bbox.dtype != np.float64:
            single_bbox = single_bbox.astype(np.float32)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(single_bbox)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0]) 
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud + Predicted 3D Bounding Boxes")

