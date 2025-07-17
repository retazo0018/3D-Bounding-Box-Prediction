import torch
from model import hybrid_3d_bbox_loss
from data_prepare import preprocess_sample
import numpy as np
from PIL import Image


def test_hybrid_loss():
    box = torch.rand(1, 25, 8, 3)  # A random box with 8 corners
    loss = hybrid_3d_bbox_loss(box, box)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Loss should be zero when inputs match, got {loss.item()}"

    box1 = torch.zeros(1, 25, 8, 3)
    box2 = torch.ones(1, 25, 8, 3)
    loss = hybrid_3d_bbox_loss(box1, box2)
    assert loss > 0, f"Loss should be positive for different boxes, got {loss.item()}"

def test_hybrid_loss_batch_shape():
    pred = torch.rand(4, 25, 8, 3)  # batch of 4
    gt = torch.rand(4, 25, 8, 3)
    loss = hybrid_3d_bbox_loss(pred, gt)
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"

def test_data_preprocessing():
    C, H, W = 3, 720, 1280
    num_instances = 5
    sample_dim = (640, 640)

    rgb_array = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
    rgb = Image.fromarray(rgb_array)
    mask = np.random.randint(0, 2, (num_instances, H, W)).astype(np.uint8)
    point_cloud = np.random.rand(3, H, W).astype(np.float32)

    rgb_tensor, mask_tensor, point_cloud_tensor = preprocess_sample(rgb, mask, point_cloud, num_instances, sample_dim)

    assert rgb_tensor.shape == (3, 640, 640), f"Invalid RGB Tensor with shape {rgb_tensor.shape}"
    assert mask_tensor.shape == (num_instances, 640, 640), f"Invalid Mask Tensor with shape {mask_tensor.shape}"
    assert point_cloud_tensor.shape == (3, 640, 640), f"Invalid Pointcloud Tensor with shape {point_cloud_tensor.shape}"
