'''
    Copyright (c) 2025 Ashwin Murali <ashwin.cse18@gmail.com>
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

import os
from PIL import Image
import torch
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
from rich.console import Console


def augment_rgb_image_without_geometry(image):
    """
    Apply non-geometric augmentations to an RGB image using Albumentations.
    """

    transform = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
        A.CLAHE(p=1.0),
    ])
    augmented = transform(image=np.array(image))

    return Image.fromarray(augmented['image'])

def preprocess_rgb(rgb, sample_dim):
    IM_H, IM_W = sample_dim
    preprocess = T.Compose([
        T.Resize((IM_H, IM_W)),
        T.ToTensor(),  # Convert HWC NumPy array to CHW PyTorch tensor
    ])
    rgb_tensor = preprocess(rgb)

    return rgb_tensor


def preprocess_sample(rgb, mask, point_cloud, num_instances, sample_dim):
    IM_H, IM_W = sample_dim

    rgb_tensor = preprocess_rgb(rgb, sample_dim)
    mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, 12, H, W)
    mask_resized = F.interpolate(mask, size=(IM_H, IM_W), mode='nearest')  # Still (1, C, IM_H, IM_W)
    mask_resized = mask_resized.squeeze() # (C, IM_H, 512)
    pad_shape = (num_instances - mask_resized.shape[0], IM_H, IM_W)
    pad_tensor = torch.zeros(pad_shape, dtype=mask_resized.dtype)
    mask_tensor = torch.cat([mask_resized, pad_tensor], dim=0)  # (25, IM_H, IM_W)

    point_cloud = torch.tensor(point_cloud).unsqueeze(0)  # (1, 3, H, W)
    point_cloud_tensor = F.interpolate(point_cloud, size=(IM_H, IM_W), mode='bilinear', align_corners=True).squeeze() # (3, IM_H, IM_W)    

    return rgb_tensor, mask_tensor, point_cloud_tensor

def load_and_prepare_data(data_dir, MAX_INSTANCES, FIXED_DIMENSION):
    rgbs, masks, pcs, bboxs3d = [], [], [], []
    IM_H, IM_W = FIXED_DIMENSION
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

                # Apply Data Augmentation on RGB Image
                augmented_rgb = preprocess_rgb(augment_rgb_image_without_geometry(rgb), FIXED_DIMENSION)
                rgb, mask, point_cloud = preprocess_sample(rgb, mask, point_cloud, num_instances=MAX_INSTANCES, sample_dim=(IM_H, IM_W))

                bbox3d_path = os.path.join(sample_path, "bbox3d.npy")
                bbox3d = np.load(bbox3d_path)
                bbox3d = pad_bounding_boxes(bbox3d, max_instances=MAX_INSTANCES)
                bbox3d = torch.from_numpy(bbox3d) # (25, 8, 3)

                rgbs.append(rgb) # (3, 512, 512)
                masks.append(mask)  # (25, 512, 512)
                pcs.append(point_cloud) # (3, 512, 512)
                bboxs3d.append(bbox3d)  # (25, 8, 3)

                rgbs.append(augmented_rgb) # (3, 512, 512)
                masks.append(mask)  # (25, 512, 512)
                pcs.append(point_cloud) # (3, 512, 512)
                bboxs3d.append(bbox3d)  # (25, 8, 3)
                
            except Exception as e:
                Console().print(f"[bold red]❌ Error loading sample from[/bold red] [white]{sample_path}[/white]: {e}", style="red")
    
    samples = {
        "rgb": rgbs,
        "mask": masks,
        "pc": pcs,
        "bbox3d": bboxs3d
    }
    
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
