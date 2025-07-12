
import torch
import numpy as np
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

class NaiveFeatureFusion(nn.Module):
    def __init__(self, rgb_feature_dim, pc_feature_dim, fused_feature_dim):
        super(NaiveFeatureFusion, self).__init__()
        self.rgb_projector = nn.Linear(rgb_feature_dim, fused_feature_dim)  
        self.pc_projector = nn.Linear(pc_feature_dim, fused_feature_dim)  
        self.fusion_layer = nn.Linear(fused_feature_dim * 2, fused_feature_dim)

    def forward(self, weighted_rgb_features, weighted_pc_features):
        B, C_r, _, _ = weighted_rgb_features.shape
        B, C_p, _, _ = weighted_pc_features.shape

        # Flatten RGB features (e.g., global average pooling)
        rgb_features_flat = weighted_rgb_features.view(B, C_r, -1).mean(dim=-1)
        pc_features_flat = weighted_pc_features.view(B, C_p, -1).mean(dim=-1)
        rgb_features_projected = self.rgb_projector(rgb_features_flat) 
        pc_features_projected = self.pc_projector(pc_features_flat) 
        
        # Concatenate both modalities
        fused_features = torch.cat([rgb_features_projected, pc_features_projected], dim=1)
        fused_features = self.fusion_layer(fused_features)

        return fused_features # (B, fused_feature_dim)

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
    def __init__(self, num_centers=25, num_proposals=25):
        super().__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.rgb_feat_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        for param in self.rgb_feat_extractor.parameters():
            param.requires_grad = False

        self.pc_feat_extractor = PCFeatureExtractor(in_channels=3, out_channels=128)
        self.feature_fusion_using_transformers = TransformerFeatureFusion(rgb_feature_dim=2048, pc_feature_dim=128, fused_feature_dim=512)
        self.center_predictor = MaskGuidedCenterPredictor(input_dim=512, num_centers=num_centers)
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

    def forward(self, batch):
        rgb, mask, pc = batch
        weighted_rgb, weighted_pc = self.weigh_features_by_instance_mask(rgb, pc, mask)
        rgb_features = self.rgb_feat_extractor(weighted_rgb)
        pc_features = self.pc_feat_extractor(weighted_pc)
        fused_features = self.feature_fusion_using_transformers.forward(rgb_features, pc_features)
        centers = self.center_predictor.forward(fused_features)
        refined_boxes = self.regressor.forward(centers)

        return refined_boxes

def hybrid_3d_bbox_loss(pred_boxes, gt_boxes):   
    pred_boxes, gt_boxes = pred_boxes.float(), gt_boxes.float()
    l2_loss = F.mse_loss(pred_boxes, gt_boxes) 

    dist = torch.cdist(pred_boxes, gt_boxes)
    chamfer_dist = torch.min(dist, dim=2)[0].mean() + torch.min(dist, dim=1)[0].mean()     

    return l2_loss+chamfer_dist

