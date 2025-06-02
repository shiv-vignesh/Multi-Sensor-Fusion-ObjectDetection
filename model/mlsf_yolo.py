import torch
import math

from .yolo import Darknet
from .yolo_utils import weights_init_normal
from trainer.loss import compute_loss

class SimpleCrossAdaptiveFusion(torch.nn.Module):
    def __init__(self, image_channels:int, lidar2d_channels:int, out_channels:int):
        
        super(SimpleCrossAdaptiveFusion, self).__init__()        
        
        in_channels = image_channels + lidar2d_channels
                
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6(inplace=True)            
        ) #3x3 
        
        in_channels = image_channels + out_channels
        self.image_transform = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=image_channels, kernel_size=1),
            torch.nn.BatchNorm2d(image_channels),
            torch.nn.ReLU6(inplace=True)
        ) #1x1
        
        in_channels = lidar2d_channels + out_channels
        self.lidar_transform = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=lidar2d_channels, kernel_size=1),
            torch.nn.BatchNorm2d(lidar2d_channels),
            torch.nn.ReLU6(inplace=True)
        ) #1x1
        
    def forward(self, image_features:torch.Tensor, lidar2d_features:torch.Tensor):
        
        concatenated_feaures = torch.concat(
            [image_features, lidar2d_features], dim=1
        )
                
        deeper_features = self.conv_layer(concatenated_feaures)

        concatenated_feaures = torch.concat(
            [image_features, deeper_features], dim=1
        )                
        
        image_features = self.image_transform(
            concatenated_feaures
        )
        
        concatenated_feaures = torch.concat(
            [lidar2d_features, deeper_features], dim=1
        )     
        
        lidar2d_features = self.lidar_transform(
            concatenated_feaures
        )   
        
        return image_features, lidar2d_features

class AttentionBasedFusion(torch.nn.Module):
    def __init__(self, image_grid_channels: int, lidar_grid_channels: int, weighted_fusion:bool,
                 alpha: float = 0.5):
        super(AttentionBasedFusion, self).__init__()
        
        # Dimension reduction for both modalities
        self.image_dim_reduce = torch.nn.Conv2d(
            image_grid_channels, image_grid_channels // 2, kernel_size=1
        )
        
        self.lidar_dim_reduce = torch.nn.Conv2d(
            lidar_grid_channels, image_grid_channels // 2, kernel_size=1
        )
        
        self.weighted_fusion = weighted_fusion

        # Cross-attention layers
        self.query_conv = torch.nn.Conv2d(image_grid_channels // 2, image_grid_channels // 2, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(image_grid_channels // 2, image_grid_channels // 2, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(image_grid_channels // 2, image_grid_channels // 2, kernel_size=1)
        
        # Output projection
        self.output_proj = torch.nn.Conv2d(
            image_grid_channels // 2, image_grid_channels, kernel_size=1
        )
        
        # Layer normalization for better training stability
        self.norm1 = torch.nn.LayerNorm([image_grid_channels // 2])
        self.norm2 = torch.nn.LayerNorm([image_grid_channels])
        
        self.dropout = torch.nn.Dropout(0.1)
        self.alpha = alpha
        self.image_grid_channels = image_grid_channels
        
        # --- Adaptive weight (gating) mechanism ---
        # This network computes two weights (for image and LiDAR) from the combined features.
        # We use global average pooling and a couple of fully connected layers.        
        if self.weighted_fusion:
            self.gate_network = torch.nn.Sequential(
                torch.nn.Conv2d(image_grid_channels * 2, 512, 1), 
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(512, 2, 1),
                torch.nn.Softmax(dim=1)  # Produces weights that sum to 1.
            )
            
        self.apply(self.weights_init_normal)
    
    @staticmethod
    def weights_init_normal(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
                          
    def forward(self, image_grid_features: torch.Tensor, lidar_grid_features: torch.Tensor):
        
        # print(image_grid_features.shape)
        
        bs, c, h, w = image_grid_features.shape

        # Dimension reduction
        image_feat = self.image_dim_reduce(image_grid_features)
        lidar_feat = self.lidar_dim_reduce(lidar_grid_features)

        # Apply layer norm
        image_feat = self.norm1(image_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        lidar_feat = self.norm1(lidar_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Compute Q, K, V
        queries = self.query_conv(image_feat)
        keys = self.key_conv(lidar_feat)
        values = self.value_conv(lidar_feat)

        # Reshape for attention
        queries = queries.view(bs, -1, h * w).permute(0, 2, 1)  # (bs, h*w, c)
        keys = keys.view(bs, -1, h * w)  # (bs, c, h*w)
        values = values.view(bs, -1, h * w).permute(0, 2, 1)  # (bs, h*w, c)

        # Compute attention scores
        attn_weights = torch.matmul(queries, keys) / math.sqrt(self.image_grid_channels // 2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        fusion_features = torch.matmul(attn_weights, values)
        fusion_features = fusion_features.permute(0, 2, 1).view(bs, -1, h, w)

        # Project back to original dimensions
        fusion_features = self.output_proj(fusion_features)
        fusion_features = self.norm2(fusion_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.weighted_fusion:

            combined_features = torch.cat([image_grid_features, lidar_grid_features], dim=1) #(bs, 2*c, gx, gy)
            weights = self.gate_network(combined_features) #(bs, 2, gx, gy)
            
            add_fusion = image_grid_features + fusion_features
            mul_fusion = image_grid_features * fusion_features
            
            image_out = add_fusion * weights[:, 0, :, :].unsqueeze(1) + mul_fusion * weights[:, 0, :, :].unsqueeze(1)
            
            add_fusion = lidar_grid_features + fusion_features
            mul_fusion = lidar_grid_features * fusion_features            
            
            lidar_out = add_fusion * weights[:, 1, :, :].unsqueeze(1) + mul_fusion * weights[:, 1, :, :].unsqueeze(1)

            return image_out, lidar_out
        
        return image_grid_features + fusion_features, lidar_grid_features + fusion_features
        
    
class CrossAdaptiveFusionModule(torch.nn.Module):
    def __init__(self, detection_depth:str, 
                image_channels:int, lidar2d_channels:int, 
                out_channels:int, fusion_type:str, 
                weighted_fusion:bool,
                num_fusion_blocks:int=5):
        
        super(CrossAdaptiveFusionModule, self).__init__()
        
        self.detection_depth = detection_depth
        
        cross_fusion_module = []
        for block in range(num_fusion_blocks):
            if fusion_type == "simple":
                fusion_module = SimpleCrossAdaptiveFusion(image_channels, 
                                                        lidar2d_channels, 
                                                        out_channels=out_channels)
                
            elif fusion_type == "attention":
                fusion_module = AttentionBasedFusion(image_grid_channels=image_channels, 
                                                    lidar_grid_channels=lidar2d_channels, 
                                                    weighted_fusion=weighted_fusion)

            cross_fusion_module.append(fusion_module)

        self.cross_fusion_module = torch.nn.Sequential(*cross_fusion_module)

        self.apply(self.weights_init_normal)
    
    @staticmethod
    def weights_init_normal(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
                
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, image_features:torch.tensor, lidar_features:torch.tensor):

        for fusion_module in self.cross_fusion_module:
            image_features, lidar_features = fusion_module(image_features, lidar_features)

        return image_features, lidar_features

class MLSFYolo(torch.nn.Module):
    def __init__(self, image_config_path:str, lidar_config_path:str,
                fusion_type:str, task_type:list,
                weighted_fusion:bool, num_fusion_blocks:int, 
                image_channels:int=3, lidar_channels:int=3,                 
                image_backbone_device:torch.device=torch.device('cpu'), 
                lidar_backbone_device:torch.device=torch.device('cpu'),
                adaptive_fusion_device:torch.device=torch.device('cpu'),
                apply_adaptive_fusion:bool=True,
                use_lidar_backbone:bool=True
                ):
        
        super(MLSFYolo, self).__init__()

        self.image_channels = image_channels
        self.lidar_channels = lidar_channels
        self.weighted_fusion = weighted_fusion

        self.image_backbone_device = image_backbone_device
        self.lidar_backbone_device = lidar_backbone_device   
        self.adaptive_fusion_device = adaptive_fusion_device

        self.apply_adaptive_fusion = apply_adaptive_fusion
        self.use_lidar_backbone = use_lidar_backbone
        self.num_fusion_blocks = num_fusion_blocks

        self.fusion_type = fusion_type
        self.task_type = task_type
        
        if "obj2d" in self.task_type and "obj_3d" in self.task_type:
            self.joint_training = True
        else:
            self.joint_training = False

        self.yolo_grid_channels = {
            '13x13':1024, 
            '26x26':512,
            '52x52':256
        }
        
        self.image_backbone = Darknet(image_config_path).to(self.image_backbone_device)
        self.image_backbone.apply(weights_init_normal)
    
        if self.use_lidar_backbone:
            self.lidar_backbone = Darknet(lidar_config_path).to(self.lidar_backbone_device)
            self.lidar_backbone.apply(weights_init_normal)

        else:
            self.lidar_backbone = None 

        self.adaptive_fusion_module = {}

        if self.apply_adaptive_fusion:
            for idx, (grid, channels) in enumerate(self.yolo_grid_channels.items()):
                self.adaptive_fusion_module[grid] = CrossAdaptiveFusionModule(
                    grid, image_channels=channels, 
                    lidar2d_channels=channels,
                    out_channels=channels, 
                    fusion_type=self.fusion_type, 
                    num_fusion_blocks=self.num_fusion_blocks, 
                    weighted_fusion=self.weighted_fusion
                )

            self.adaptive_fusion_module = torch.nn.ModuleDict(self.adaptive_fusion_module).to(self.adaptive_fusion_device)

    def forward(self, images:torch.tensor, lidar_2d:torch.tensor=None, 
                targets:torch.tensor=None, targets_3d:torch.Tensor=None, 
                cls_loss_type:str="focal"):

        assert images is not None or lidar_2d is not None, f"Both Modalities: images and lidar cannot be {images} {lidar_2d}"

        image_backbone_features = self.image_backbone.forward_backbone(
            images.to(self.image_backbone_device)
        ) if images is not None else None

        lidar_2d_features = self.lidar_backbone.forward_backbone(
            lidar_2d.to(self.lidar_backbone_device)
        ) if self.use_lidar_backbone and lidar_2d is not None  else None

        if self.apply_adaptive_fusion and lidar_2d_features is not None and image_backbone_features is not None:        
            fused_features_list = []
            for idx, (_, fusion_module) in enumerate(self.adaptive_fusion_module.items()):
                image_feat, lidar_feat = fusion_module(image_backbone_features[idx].to(self.adaptive_fusion_device), 
                                            lidar_2d_features[idx].to(self.adaptive_fusion_device))

                fused_features = self.combine_features(image_feat, lidar_feat)
                fused_features_list.append(fused_features.to(self.image_backbone_device))

        elif lidar_2d_features is not None and image_backbone_features is not None:
            fused_features_list = []
            for idx, (image_feat, lidar_feat) in enumerate(zip(image_backbone_features, lidar_2d_features)):
                fused_features = self.combine_features(image_feat, lidar_feat.to(image_feat.device))
                fused_features_list.append(fused_features.to(self.image_backbone_device))

        elif image_backbone_features is None:
            fused_features_list = []
            for idx, lidar_feat in enumerate(lidar_2d_features):
                fused_features_list.append(lidar_feat.to(self.image_backbone_device))   

        elif lidar_2d_features is None:
            fused_features_list = []
            for idx, image_feat in enumerate(image_backbone_features):
                fused_features_list.append(image_feat.to(self.image_backbone_device))

        loss_components = {}
        total_loss = 0.0
        task_outputs = {}

        if "obj_2d" in self.task_type:
            outputs = self.image_backbone.forward_detection_head(
                fused_features_list, images.shape[2] if images is not None else lidar_2d.shape[2]
            )

            if targets is not None:                
                if not self.training:
                    num_anchors = 3
                    for i, x in enumerate(outputs):
                        bs, num_preds, _ = x.shape
                        grid_size = int(math.sqrt(num_preds // num_anchors))
                        outputs[i] = x.view(bs, num_anchors, grid_size, grid_size, -1)          

                if torch.is_tensor(targets):                    
                    loss_2d, loss_components_2d = compute_loss(outputs,
                                                        targets.to(self.image_backbone_device), 
                                                        self.image_backbone, cls_loss_type=cls_loss_type)
                    loss_components['obj_2d'] = {
                                'lbox':loss_components_2d[0].item(), 
                                'lobj':loss_components_2d[1].item(),
                                'lcls':loss_components_2d[2].item()
                                }                    
                    
                else:
                    loss_cls = torch.tensor(0.0, device=self.image_backbone_device, requires_grad=True)
                    loss_obj = torch.tensor(0.0, device=self.image_backbone_device, requires_grad=True)
                    loss_box = torch.tensor(0.0, device=self.image_backbone_device, requires_grad=True)
                    
                    loss_2d = loss_cls + loss_obj + loss_box
                    loss_components['obj_2d'] = {
                                'lbox':loss_box.item(), 
                                'lobj':loss_obj.item(),
                                'lcls':loss_cls.item()
                                }                        

                task_outputs['obj_2d'] = outputs
                # total_loss += loss_2d

            else:
                if not self.training:
                    num_anchors = 3                    
                    for i, x in enumerate(outputs):
                        bs, num_preds, _ = x.shape
                        grid_size = int(math.sqrt(num_preds // num_anchors))
                        outputs[i] = x.view(bs, num_anchors, grid_size, grid_size, -1)
                        
                    task_outputs['obj_2d'] = outputs

        if "obj_3d" in self.task_type:
            if self.lidar_backbone.has_3d_head:
                outputs, loss_3d, metrics_all, cls_loss_3d, bbox_loss_3d, conf_loss_3d = self.lidar_backbone.forward_detection_head_3d(
                    lidar_2d_features, 
                    lidar_2d.shape[2] if images is not None else lidar_2d.shape[2], 
                    targets_3d.to(self.lidar_backbone_device))
                
                loss_components['obj_3d'] = metrics_all 
                task_outputs['obj_3d'] = outputs
                
                # total_loss += loss_3d
                
            else:
                raise Exception('Cannot Compute 3D Object Offsets without 3D Head for lidar Backbone')

        if self.joint_training and (targets is not None and targets_3d is not None):
            loss_cls = cls_loss_3d + loss_components['obj_2d']['lcls']
            loss_box = bbox_loss_3d + loss_components['obj_2d']['lbox']
            loss_conf = conf_loss_3d + loss_components['obj_2d']['lobj']
            total_loss = loss_cls + loss_box + loss_conf

            return total_loss, loss_components, task_outputs
        
        else:
            if "obj_3d" in self.task_type:
                total_loss += loss_3d
            
            elif "obj_2d" in self.task_type:
                total_loss += loss_2d
        
        return total_loss, loss_components, task_outputs
    
    def combine_features(self, img_feat:torch.Tensor, lidar_feat:torch.Tensor=None):
        
        if lidar_feat is not None:
            # Normalize each feature map along dim=1
            image_mean = img_feat.mean(dim=1, keepdim=True)
            lidar_mean = lidar_feat.mean(dim=1, keepdim=True)

            image_norm = img_feat - image_mean
            lidar_norm = lidar_feat - lidar_mean

            # Fuse after normalization
            return (image_norm + lidar_norm)
            
        else:
            return img_feat
        
        
