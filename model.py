
import torch
import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F  
import torchvision.models as models
from torchvision.transforms.functional import resize


class BBoxRegressor(nn.Module):
    def __init__(self, input_dim, num_proposals):
        super().__init__()
        self.num_proposals = num_proposals
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_proposals * 24)
        )

    def forward(self, roi_features):

        return self.fc(roi_features).view(-1, self.num_proposals, 8, 3) # (B, num_proposals, 8, 3)

class MaskGuidedCenterPredictor(nn.Module):
    def __init__(self, input_dim, num_centers):
        super().__init__()
        self.num_centers = num_centers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_centers * 3)  # Predict [x, y, z] for each center
        )

    def forward(self, mask_features):

        return self.fc(mask_features) # (B, num_centers*3)

class FeatureFusion(nn.Module):
    def __init__(self, rgb_feature_dim, pc_feature_dim, fused_feature_dim):
        super(FeatureFusion, self).__init__()
        self.rgb_projector = nn.Linear(rgb_feature_dim, fused_feature_dim)  
        self.pc_projector = nn.Linear(pc_feature_dim, fused_feature_dim)  
        self.fusion_layer = nn.Linear(fused_feature_dim * 2, fused_feature_dim)

    def forward(self, weighted_rgb_features, weighted_pc_features):
        B, C, H, W = weighted_rgb_features.shape
        B, num_points, pc_feature_dim = weighted_pc_features.shape

        # Flatten RGB features (e.g., global average pooling)
        rgb_features_flat = weighted_rgb_features.view(B, C, -1).mean(dim=-1)
        rgb_features_projected = self.rgb_projector(rgb_features_flat) 
        pc_features_projected = self.pc_projector(weighted_pc_features.mean(dim=1).to(torch.float32))  
        
        # Concatenate both modalities
        fused_features = torch.cat([rgb_features_projected, pc_features_projected], dim=1)
        fused_features = self.fusion_layer(fused_features)
        return fused_features # (B, fused_feature_dim)


class MultiObject3DBBoxModel(nn.Module):
    def __init__(self, num_centers=20, num_proposals=25):
        super().__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.rgb_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        self.feature_fusion = FeatureFusion(rgb_feature_dim=2048, pc_feature_dim=33, fused_feature_dim=512)
        self.center_predictor = MaskGuidedCenterPredictor(input_dim=512, num_centers=num_centers)
        self.regressor = BBoxRegressor(input_dim=num_centers*3, num_proposals=num_proposals)

    def weigh_features_by_instance_mask(self, rgb_features, pc_features, mask):
        B, C, H_feat, W_feat = rgb_features.shape
        _, H_mask, W_mask = mask.shape

        # Resize all masks to the feature map size for each batch
        resized_mask_rgb = torch.zeros((B, H_feat, W_feat), dtype=torch.float32, device=mask.device)
        for b in range(B):
            resized_mask = resize(mask[b].unsqueeze(0), (H_feat, W_feat))
            resized_mask_rgb[b] = resized_mask

        # Weight RGB features using combined masks
        resized_mask_rgb = resized_mask_rgb.unsqueeze(1)     # (B, H, W) - > (B, 1, H, W)
        weighted_rgb_features = rgb_features * resized_mask_rgb  # Shape: (B, C, H_feat, W_feat)
        
        # Resize masks to point cloud dimensions and compute weights
        point_weights_batch = []
        for b in range(B):     
            pc_features_b = pc_features[b]    
            # Flatten combined mask and compute weights
            point_indices = np.arange(pc_features_b.shape[0]).reshape(H_mask, W_mask).flatten()
            point_weights = mask[b].flatten()[point_indices]
            point_weights = point_weights / (torch.max(point_weights) + 1e-8)
            point_weights_batch.append(point_weights.unsqueeze(1))
        
        # Stack point weights and weight point cloud features
        point_weights_batch = torch.stack(point_weights_batch)  
        weighted_pc_features = pc_features * point_weights_batch

        return weighted_rgb_features, weighted_pc_features #  (B, num_points, feature_dim)

    def forward(self, batch):
        rgb, mask, pc_features = batch
        rgb_features = self.rgb_extractor(rgb)
        weighted_rgb_features, weighted_pc_features = self.weigh_features_by_instance_mask(rgb_features, pc_features, mask)
        fused_features = self.feature_fusion.forward(weighted_rgb_features, weighted_pc_features)
        centers = self.center_predictor.forward(fused_features)
        refined_boxes = self.regressor.forward(centers)

        return refined_boxes

def hybrid_3d_bbox_loss(pred_corners, gt_corners):   
    pred_corners, gt_corners = pred_corners.float(), gt_corners.float()
    l2_loss = F.mse_loss(pred_corners, gt_corners) 
    dist = torch.cdist(pred_corners, gt_corners)
    chamfer_dist = torch.min(dist, dim=3)[0].mean() + torch.min(dist, dim=2)[0].mean()    

    return l2_loss+chamfer_dist

