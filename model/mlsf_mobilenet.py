from itertools import chain
from typing import List, Tuple

import torch
from torchvision.models.mobilenet import mobilenet_v2

from .mlsf_yolo import CrossAdaptiveFusionModule
from trainer.loss import compute_loss

class DetectionHead(torch.nn.Module):
    def __init__(self, in_channels:int, num_classes:int, 
                anchors:List[Tuple[int, int]]):
        
        super(DetectionHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.no = 5 + self.num_classes
        self.out_channels = self.num_anchors * (self.num_classes + 5)

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))        
        self.stride = None
        
        self.downsampling = torch.nn.Conv2d(
            self.in_channels, 
            self.out_channels,
            kernel_size=1, 
            stride=1
        )
        
        self.apply(self.weights_init_normal)
        
    @staticmethod
    def weights_init_normal(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x:torch.tensor, img_size:int):
        
        x = self.downsampling(x)

        bs, dim, ny, nx = x.shape
        self.stride = img_size // x.shape[2]
        
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
        # if not self.training:
        #     x = x.view(bs, -1, self.no)
            
        return x

class MLSFMobilenet(torch.nn.Module):
    def __init__(self, image_channels:int=3, lidar_channels:int=3, 
                fusion_type:str="attention", num_fusion_blocks:int=2, 
                num_classes:int=3, weighted_fusion:bool=False,
                image_backbone_device:torch.device=torch.device('cpu'), 
                lidar_backbone_device:torch.device=torch.device('cpu'),
                adaptive_fusion_device:torch.device=torch.device('cpu'),
                detection_depths:str=['InvertedResidual_6', 'InvertedResidual_12', 'InvertedResidual_17'],
                apply_adaptive_fusion:bool=True,
                use_lidar_backbone:bool=True,
                pretrained_backbone:bool=True, 
                image_resize:tuple=(608, 608), 
                task_type=['obj_2d']
                ):
        
        super(MLSFMobilenet, self).__init__()
        
        self.image_channels = image_channels
        self.lidar_channels = lidar_channels     
        self.num_classes = num_classes

        self.image_backbone_device = image_backbone_device
        self.lidar_backbone_device = lidar_backbone_device   

        self.detection_depths = detection_depths
        self.adaptive_fusion_device = adaptive_fusion_device
        
        self.apply_adaptive_fusion = apply_adaptive_fusion
        self.use_lidar_backbone = use_lidar_backbone
        
        self.fusion_type = fusion_type
        self.num_fusion_blocks = num_fusion_blocks
        self.weighted_fusion = weighted_fusion
        self.image_resize = image_resize
        self.image_width, self.image_height = image_resize
        self.task_type = task_type
        
        if pretrained_backbone:
            self.image_backbone = mobilenet_v2(pretrained=True).features
            self.image_backbone.to(self.image_backbone_device)
            
            if self.use_lidar_backbone:
                self.lidar_backbone = mobilenet_v2(pretrained=True).features
                self.lidar_backbone.to(self.lidar_backbone_device) 
            else:
                self.lidar_backbone = None
        
        self.depth_channels = {
            'InvertedResidual_6':32,
            'InvertedResidual_7':64,
            'InvertedResidual_11':96,
            'InvertedResidual_12':96,
            'InvertedResidual_15':160,
            'InvertedResidual_17':320,
            'InvertedResidual_18':1280
        }
        
        if self.image_resize == (608, 608):

            self.anchor_info = {
                'InvertedResidual_6':[(15,34), (39,57), (43,144)], #(76x76) #small objects
                'InvertedResidual_12':[(81,290), (82,81), (132,146)], # (38x38) medium objects
                'InvertedResidual_17':[(149,549), (182,273), (344,354)] #(19x19) large objects
            }

            #kitti anchors
            # self.anchor_info = {
            #     'InvertedResidual_6':[(12,47), (20,110), (30,195)], #(76x76) #small objects
            #     'InvertedResidual_12':[(44,70), (49,438), (51,294)], # (38x38) medium objects
            #     'InvertedResidual_17':[(89,149), (94,639), (94,429)] #(19x19) large objects
            # }
            
        if self.image_resize == (416, 416):
            self.anchor_info = {
                'InvertedResidual_6':[(61,110), (62,309), (63,441)], #(52x52) #small objects
                'InvertedResidual_12':[(27,167), (31,48), (37,247)], # (26x26) medium objects
                'InvertedResidual_17':[(8,29), (12,66), (18,112)] #(13x13) large objects
            }
            
        if self.image_resize == (320, 320):
            self.anchor_info = {
                'InvertedResidual_6':  [(7,25), (11,59), (17,105)],  # 40x40 → small objects
                'InvertedResidual_12':[(24,37), (26,167), (42,331)],  # 20x20 → medium objects
                'InvertedResidual_17':[(42,229), (47,84), (85,305)] # 10x10 → large objects
            } 
            
        if self.image_resize == (224, 224):
            self.anchor_info = {
                'InvertedResidual_6':  [(5,17), (7,42), (9,259)],  # 28x28 → small objects
                'InvertedResidual_12':[(12,74), (17,26), (18,117)],  # 14x14 → medium objects
                'InvertedResidual_17':[(30,162), (33,59), (39,226)] # 7x7 → large objects
            }

        self.adaptive_fusion_module = {} 
        self.detection_heads = {}        
        
        if self.apply_adaptive_fusion:
            for depth in self.detection_depths:
                self.adaptive_fusion_module[depth] = CrossAdaptiveFusionModule(
                    depth, 
                    image_channels=self.depth_channels[depth], 
                    lidar2d_channels=self.depth_channels[depth], 
                    out_channels=self.depth_channels[depth],
                    fusion_type=self.fusion_type, 
                    num_fusion_blocks=self.num_fusion_blocks, 
                    weighted_fusion=self.weighted_fusion
                )
            
            self.adaptive_fusion_module = torch.nn.ModuleDict(self.adaptive_fusion_module).to(self.adaptive_fusion_device)
            
        for depth in self.detection_depths:
            self.detection_heads[depth] = DetectionHead(
                self.depth_channels[depth], 
                self.num_classes, 
                self.anchor_info[depth]
            )

        self.detection_heads = torch.nn.ModuleDict(self.detection_heads).to(self.adaptive_fusion_device)

    def get_features(self, backbone:torch.nn.Module, x:torch.Tensor):
        features = {}
        hook_handles = []

        # Define a closure to capture the layer name
        def get_hook(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Iterate over backbone's children (for MobileNetV2 features, keys are strings like '0', '1', ...)
        for name, module in backbone.named_children():
            if f'{module.__class__.__name__}_{name}' in self.detection_depths:
                name = f'{module.__class__.__name__}_{name}'
                handle = module.register_forward_hook(get_hook(name))
                hook_handles.append(handle)

        # Forward pass through the backbone; hooks will capture the outputs.
        _ = backbone(x)

        # Remove hooks to prevent side effects on subsequent passes.
        for handle in hook_handles:
            handle.remove()

        # for depth, feature in features.items():
        #     features[depth] = torch.nn.LeakyReLU(0.1)(feature)

        return features
    
    def forward(self, images:torch.tensor, lidar_2d:torch.tensor=None, 
                targets:torch.tensor=None, targets_3d:torch.tensor=None,
                cls_loss_type:str="focal"
                ):
        
        image_features = self.get_features(
            self.image_backbone, images.to(self.image_backbone_device)
        )
        
        lidar_2d_features = self.get_features(
            self.lidar_backbone, lidar_2d.to(self.lidar_backbone_device)
        ) if lidar_2d is not None and self.use_lidar_backbone else None

        if self.apply_adaptive_fusion and lidar_2d_features is not None and image_features is not None:        
            fused_features_list = []
            for idx, (depth, fusion_module) in enumerate(self.adaptive_fusion_module.items()):
                
                image_feat, lidar_feat = fusion_module(image_features[depth].to(self.adaptive_fusion_device), 
                                            lidar_2d_features[depth].to(self.adaptive_fusion_device))

                fused_features = self.combine_features(image_feat, lidar_feat)
                fused_features_list.append(fused_features.to(self.image_backbone_device))

        elif lidar_2d_features is not None and image_features is not None:
            fused_features_list = []
            for idx, (image_feat, lidar_feat) in enumerate(zip(image_features, lidar_2d_features)):
                fused_features = self.combine_features(image_feat, lidar_feat.to(image_feat.device))
                fused_features_list.append(fused_features.to(self.image_backbone_device))

        else:
            fused_features_list = [image_feature for image_feature in image_features.values()]
        
        for idx, (depth, detection_head) in enumerate(self.detection_heads.items()):
            fused_features_list[idx] = detection_head(
                fused_features_list[idx].to(self.adaptive_fusion_device), 
                images.shape[2]
            )

        if targets is not None:

            loss_2d, loss_2d_components = compute_loss(
                fused_features_list, 
                targets.to(self.adaptive_fusion_device), 
                model=self.detection_heads,
                model_type='mobilenet',
                cls_loss_type=cls_loss_type
            )

            loss_2d_components = {
                        'lbox':loss_2d_components[0].item(), 
                        'lobj':loss_2d_components[1].item(),
                        'lcls':loss_2d_components[2].item()
                        }

            return loss_2d, {"obj_2d":loss_2d_components}, {"obj_2d":fused_features_list}

        else:
            return 0.0, {}, {
                'obj_2d': fused_features_list
            }

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
    