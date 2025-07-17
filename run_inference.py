
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from data_prepare import preprocess_sample, pad_bounding_boxes

if __name__=="__main__":
    model = torch.load('model.pt')
    file_name = "8b061a8b-9915-11ee-9103-bbb8eae05561"
    rgb_path = f"./data/{file_name}/rgb.jpg"
    mask_path = f"./data/{file_name}/mask.npy"
    bbox3d_path = f"./data/{file_name}/bbox3d.npy"
    pc_path = f"./data/{file_name}/pc.npy"

    rgb = Image.open(rgb_path).convert('RGB')
    mask = np.load(mask_path)
    point_cloud = np.load(pc_path)
    ground_bbox = np.load(bbox3d_path)
    ground_bbox = pad_bounding_boxes(ground_bbox, max_instances=25)
    rgb, mask, point_cloud = preprocess_sample(rgb, mask, point_cloud, num_instances=25, sample_dim=(512, 512))

    pred_box = model(rgb.unsqueeze(0), mask.unsqueeze(0), point_cloud.unsqueeze(0))
    pred_box = pred_box[0].detach().numpy()

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(point_cloud.reshape(3, -1).T)
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

