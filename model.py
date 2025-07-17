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

import torch
import torch.nn as nn
import torch.nn.functional as F  
import torchvision.models as models


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

class CenterPredictor(nn.Module):
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

class TransformerFeatureFusion(nn.Module):
    def __init__(self, rgb_feature_dim, pc_feature_dim, fused_feature_dim, num_heads=4, num_layers=2):
        super(TransformerFeatureFusion, self).__init__()

        self.rgb_projector = nn.Linear(rgb_feature_dim, fused_feature_dim)
        self.pc_projector = nn.Linear(pc_feature_dim, fused_feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_feature_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # to pool token dimension
        self.output_layer = nn.Linear(fused_feature_dim, fused_feature_dim)

    def forward(self, weighted_rgb_features, weighted_pc_features):
        B, C_r, H_r, W_r = weighted_rgb_features.shape
        B, C_p, H_p, W_p = weighted_pc_features.shape

        # Global average pool: (B, C)
        rgb_features_flat = weighted_rgb_features.view(B, C_r, -1).mean(dim=-1)
        pc_features_flat = weighted_pc_features.view(B, C_p, -1).mean(dim=-1)

        # Project to shared dimension: (B, D)
        rgb_projected = self.rgb_projector(rgb_features_flat)
        pc_projected = self.pc_projector(pc_features_flat)

        # Stack features as tokens: (B, 2, D)
        tokens = torch.stack([rgb_projected, pc_projected], dim=1)

        # Transformer encoder: (B, 2, D) → (B, 2, D)
        fused_tokens = self.transformer_encoder(tokens)

        # Pool over tokens (e.g., mean): (B, D)
        fused_features = fused_tokens.mean(dim=1)

        # Output layer
        fused_features = self.output_layer(fused_features)

        return fused_features  # (B, fused_feature_dim)

class PCFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, pc):  # pc: [B, 3, H_feat, W_feat]
        return self.extractor(pc)  # [B, out_channels, H_feat, W_feat]

class MultiObject3DBBoxModel(nn.Module):
    def __init__(self, MAX_INSTANCES):
        super().__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.rgb_feat_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        for param in self.rgb_feat_extractor.parameters():
            param.requires_grad = False
        num_centers, num_proposals = MAX_INSTANCES, MAX_INSTANCES

        self.pc_feat_extractor = PCFeatureExtractor(in_channels=3, out_channels=128)
        self.feature_fusion_using_transformers = TransformerFeatureFusion(rgb_feature_dim=2048, pc_feature_dim=128, fused_feature_dim=512)
        self.center_predictor = CenterPredictor(input_dim=512, num_centers=num_centers)
        self.regressor = BBoxRegressor(input_dim=num_centers*3, num_proposals=num_proposals)

    def weigh_features_by_instance_mask(self, rgb, pc, mask):
        # Resize all masks to the feature map size for each batch
        resized_mask = mask.max(dim=1).values  # Shape: (B, H, W)

        # Make a soft attention mask
        attention_map = torch.where(
            resized_mask > 0,
            torch.tensor(1.5, device=mask.device),
            torch.tensor(1.0, device=mask.device)
        )  # (B, H, W)
        attention_map = attention_map.unsqueeze(1)  # (B, 1, H, W)  

        # Weight RGB features using combined masks
        weighted_rgb = rgb * attention_map  # Shape: (B, C, H, W)
        
        # Weight PC using combined mask
        weighted_pc = pc * attention_map
        weighted_pc = weighted_pc.float()
        weighted_pc = F.adaptive_avg_pool2d(weighted_pc, (16, 16)) # Shape: (B, 3, H, W)

        return weighted_rgb, weighted_pc

    def forward(self, rgb, mask, pc):
        weighted_rgb, weighted_pc = self.weigh_features_by_instance_mask(rgb, pc, mask)
        rgb_features = self.rgb_feat_extractor(weighted_rgb)
        pc_features = self.pc_feat_extractor(weighted_pc)
        fused_features = self.feature_fusion_using_transformers.forward(rgb_features, pc_features)
        centers = self.center_predictor.forward(fused_features)
        refined_boxes = self.regressor.forward(centers)

        return refined_boxes

def chamfer_distance_single_box(pred_corners, gt_corners):
    # pred_corners, gt_corners: (8, 3)
    dist = torch.cdist(pred_corners.unsqueeze(0), gt_corners.unsqueeze(0))  # (1, 8, 8)
    min_dist1 = dist.min(dim=2)[0].mean()  # pred -> gt
    min_dist2 = dist.min(dim=1)[0].mean()  # gt -> pred
    return min_dist1 + min_dist2

def hybrid_3d_bbox_loss(pred_boxes, gt_boxes):   
    pred_boxes = pred_boxes.float()  # (B, N, 8, 3)
    gt_boxes = gt_boxes.float()
    l2 = F.mse_loss(pred_boxes, gt_boxes)

    chamfer = 0.0
    B, N, _, _ = pred_boxes.shape
    for b in range(B):
        for n in range(N):
            chamfer += chamfer_distance_single_box(pred_boxes[b, n], gt_boxes[b, n])
    chamfer = chamfer / (B * N)
    
    return l2 + 0.5*chamfer
