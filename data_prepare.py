
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F


def preprocess_sample(rgb, mask, point_cloud, num_instances, sample_dim):
    IM_H, IM_W = sample_dim

    preprocess = T.Compose([
        T.Resize((IM_H, IM_W)),
        T.ToTensor(),  # Convert HWC NumPy array to CHW PyTorch tensor
    ])
    rgb_tensor = preprocess(rgb)

    mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, 12, H, W)
    mask_resized = F.interpolate(mask, size=(IM_H, IM_W), mode='nearest')  # Still (1, C, 512, 512)
    mask_resized = mask_resized.squeeze() # (C, 512, 512)
    pad_shape = (num_instances - mask_resized.shape[0], IM_H, IM_W)
    pad_tensor = torch.zeros(pad_shape, dtype=mask_resized.dtype)
    mask_tensor = torch.cat([mask_resized, pad_tensor], dim=0)  # (25, 512, 512)

    point_cloud = torch.tensor(point_cloud).unsqueeze(0)  # (1, 3, H, W)
    point_cloud_tensor = F.interpolate(point_cloud, size=(IM_H, IM_W), mode='bilinear', align_corners=True).squeeze() # (3, 512, 512)
    
    return rgb_tensor, mask_tensor, point_cloud_tensor

def load_data(data_dir):
    rgbs, masks, pcs, bboxs3d = [], [], [], []
    MAX_INSTANCES = 25
    IM_H, IM_W = 512, 512
    
    for sample_dir in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, sample_dir)
        if os.path.isdir(sample_path):
            try:
                rgb_path = os.path.join(sample_path, "rgb.jpg")
                rgb = Image.open(rgb_path).convert('RGB')
                mask_path = os.path.join(sample_path, "mask.npy")
                mask = np.load(mask_path)
                pc_path = os.path.join(sample_path, "pc.npy")
                point_cloud = np.load(pc_path)
                rgb, mask, point_cloud = preprocess_sample(rgb, mask, point_cloud, num_instances=MAX_INSTANCES, sample_dim=(IM_H, IM_W))

                bbox3d_path = os.path.join(sample_path, "bbox3d.npy")
                bbox3d = np.load(bbox3d_path)
                bbox3d = pad_bounding_boxes(bbox3d, max_instances=MAX_INSTANCES)
                bbox3d = torch.from_numpy(bbox3d) # (25, 8, 3)

                rgbs.append(rgb) # (3, 512, 512)
                masks.append(mask)  # (25, 512, 512)
                pcs.append(point_cloud) # (3, 512, 512)
                bboxs3d.append(bbox3d)  # (25, 8, 3)

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
        point_cloud = self.point_cloud[idx]  
        bbox3d = self.bbox3d[idx]
        mask = self.mask[idx]

        return rgb, mask, point_cloud, bbox3d

def pad_bounding_boxes(bbox_data, max_instances):
    # If the number of instances is less than max_instances, pad with zeros
    if bbox_data.shape[0] < max_instances:
        padding = np.zeros((max_instances - bbox_data.shape[0], 8, 3))
        bbox_data = np.vstack([bbox_data, padding])
    
    return bbox_data
