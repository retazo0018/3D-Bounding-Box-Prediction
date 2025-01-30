
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pc_feature_extractor(pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        fpfh_features = torch.from_numpy(np.array(fpfh.data).T) 

        return fpfh_features # (N, 33)

def preprocess_image(rgb_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert HWC NumPy array to CHW PyTorch tensor
    ])
    rgb_tensor = preprocess(rgb_image)
    
    return rgb_tensor # Shape: (3, H, W)

def load_data(data_dir):
    rgbs, masks, pcs, bboxs3d = [], [], [], []
    for sample_dir in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, sample_dir)
        if os.path.isdir(sample_path):
            try:
                rgb_path = os.path.join(sample_path, "rgb.jpg")
                rgb = cv2.imread(rgb_path)
                o_h, o_w = rgb.shape[0], rgb.shape[1]
                rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
                rgb = preprocess_image(rgb)
                rgbs.append(rgb)

                mask_path = os.path.join(sample_path, "mask.npy")
                mask = np.load(mask_path)
                mask = np.any(mask, axis=0, keepdims=True).squeeze()
                mask_uint8 = mask.astype(np.uint8)
                mask = A.Resize(480, 640)(image=mask_uint8)["image"]
                mask = torch.from_numpy(mask)
                masks.append(mask)
                
                pc_features_path = os.path.join(sample_path, "pc_features.npz")
                if os.path.exists(pc_features_path):
                    print(f"Loaded data from {pc_features_path}.")
                    pc_features = torch.from_numpy(np.load(pc_features_path)['data_array'])
                else:
                    pc_path = os.path.join(sample_path, "pc.npy")
                    pc = np.load(pc_path).reshape(3, -1).T
                    pc_reshaped = pc.reshape(o_h, o_w, 3)
                    resized_x = A.Resize(480, 640)(image=pc_reshaped[:, :, 0])["image"]
                    resized_y = A.Resize(480, 640)(image=pc_reshaped[:, :, 1])["image"]
                    resized_z = A.Resize(480, 640)(image=pc_reshaped[:, :, 2])["image"]
                    pc = np.stack([resized_x, resized_y, resized_z], axis=-1).reshape(-1, 3)
                    pc_features = np.asarray(pc_feature_extractor(pc))
                    print(f"Saved pc_features at {pc_features_path}.")
                    np.savez_compressed(pc_features_path, data_array=pc_features)
                    pc_features = torch.from_numpy(pc_features)
                pcs.append(pc_features)

                bbox3d_path = os.path.join(sample_path, "bbox3d.npy")
                bbox3d = np.load(bbox3d_path)
                bbox3d = pad_bounding_boxes(bbox3d)
                bbox3d = torch.from_numpy(bbox3d)
                bboxs3d.append(bbox3d)

            except Exception as e:
                print(f"Error loading sample from {sample_path}: {e}")
    
    samples = {
        "rgb": rgbs,
        "mask": masks,
        "pc": pcs,
        "bbox3d": bboxs3d
    }
    
    return samples

def prepare_data(data_dir):
    samples = load_data(data_dir)
    print(f"Loaded {len(samples['rgb'])} samples.")

    return samples

class Custom3DBBoxDataset(Dataset):
    def __init__(self, data):
        self.rgb = data['rgb']  
        self.mask = data['mask']  
        self.point_cloud = data['pc']  
        self.bbox3d = data['bbox3d']  

    def __len__(self):

        return len(self.rgb)

    def __getitem__(self, idx):
        rgb = self.rgb[idx]
        mask = self.mask[idx]
        point_cloud = self.point_cloud[idx]  
        bbox3d = self.bbox3d[idx]

        return rgb, mask, point_cloud, bbox3d

def pad_bounding_boxes(bbox_data, max_instances=25):
    # If the number of instances is less than max_instances, pad with zeros
    if bbox_data.shape[0] < max_instances:
        padding = np.zeros((max_instances - bbox_data.shape[0], 8, 3))
        bbox_data = np.vstack([bbox_data, padding])
    
    return bbox_data
