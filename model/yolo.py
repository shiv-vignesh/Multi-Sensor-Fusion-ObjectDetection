from __future__ import division

import os, math
from itertools import chain
from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_utils import parse_model_config
from trainer.loss import build_targets_3d


def create_modules(module_defs: List[dict]) -> Tuple[dict, nn.ModuleList]:
    """
    Constructs module list of layer blocks from module configuration in module_defs

    :param module_defs: List of dictionaries with module definitions
    :return: Hyperparameters and pytorch module list
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", nn.Mish())
            elif module_def["activation"] == "logistic":
                modules.add_module(f"sigmoid_{module_i}", nn.Sigmoid())
            elif module_def["activation"] == "swish":
                modules.add_module(f"swish_{module_i}", nn.SiLU())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            new_coords = bool(module_def.get("new_coords", False))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, new_coords)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        elif module_def['type'] == 'yolo_3d':
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [float(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1], math.sin(anchors[i + 2]), math.cos(anchors[i + 2])) for i in range(0, len(anchors), 3)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            
            yolo_layer = YOLOLayer3D(anchors, num_classes)
            modules.add_module(f"yolo3d_{module_i}", yolo_layer)            
        
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode: str = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer3D_2(nn.Module):
    
    def __init__(self, anchors: List[Tuple[float, float, float]], num_classes: int):
        super(YOLOLayer3D_2, self).__init__()
        
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        
        # x, y, w, l, sin(yaw), cos(yaw), objectness, class_logits
        self.no = num_classes + 7 # 7 regression + classification
        # Register anchors and anchor grid
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 4)  # [w, l, im, re]
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 4))

        self.stride = None  # Will be computed dynamically
        self.grid_size = 0
        
    def compute_grid_offsets(self, grid_size:int, device:torch.device):
        """
        - Computes grid_size
        - creates gridx and gridy matrix
        - scales anchors to grid dimension
        - computes anchor_w and anchor_h corresponding to grid dimension
        """
        
        self.grid_size = grid_size                
        self.grid_x = torch.arange(self.grid_size).repeat(self.grid_size, 1).view([1, 1, self.grid_size, self.grid_size]).float().to(device)
        self.grid_y = torch.arange(self.grid_size).repeat(self.grid_size, 1).t().view([1, 1, self.grid_size, self.grid_size]).float().to(device)
        
        self.scaled_anchors = self.anchors.clone()
        self.scaled_anchors[:, :2] /= self.stride
        
        self.anchor_w = self.scaled_anchors[:, 0].view(1, self.num_anchors, 1, 1)
        self.anchor_h = self.scaled_anchors[:, 1].view(1, self.num_anchors, 1, 1)

    def forward(self, x:torch.tensor, img_size:int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor (B, (no * num_anchors), H, W)
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2)
        self.stride = stride
        
        bs, _, ny, nx = x.shape
        x = x.view(bs, self.num_anchors, self.no, ny, nx)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        
        grid_size = x.shape[2]
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, x.device)
        
        if not self.training:
            x = x.view(bs, -1, self.no)
            return x

        return x

class YOLOLayer3D(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=None):
        super(YOLOLayer3D, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        
        self.use_iou_loss = False
        self.use_focal_loss = True

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride, im, re) for a_w, a_h, im, re in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, img_dim=None, targets=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        im = prediction[..., 4]  # angle imaginary part
        re = prediction[..., 5]  # angle real part
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :6].shape).to(prediction.device)
        pred_boxes[..., 0] = x.data + self.grid_x.to(prediction.device)
        pred_boxes[..., 1] = y.data + self.grid_y.to(prediction.device)
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w.to(prediction.device)
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h.to(prediction.device)
        pred_boxes[..., 4] = im
        pred_boxes[..., 5] = re

        output = torch.cat(
            (
                #pred_boxes.view(num_samples, -1, 6) * self.stride,
                pred_boxes[..., :4].view(num_samples, -1, 4) * self.stride,
                pred_boxes[..., 4:].view(num_samples, -1, 2),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0, self.metrics
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf = build_targets_3d(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors.to(prediction.device),
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            
            if self.use_iou_loss:
                loss_box = (1.0 - iou_scores).mean()               
            
            else:
                loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
                loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
                loss_im = self.mse_loss(im[obj_mask], tim[obj_mask])
                loss_re = self.mse_loss(re[obj_mask], tre[obj_mask])
                loss_eular = loss_im + loss_re
                
                loss_box = loss_x + loss_y + loss_w + loss_h + loss_eular

            # loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            # loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            tobj = torch.zeros_like(pred_conf, device=pred_conf.device)
            tobj = iou_scores
            loss_conf = self.bce_loss(pred_conf, tobj.clamp(0))         
        
            if self.use_focal_loss:
                alpha=0.25
                gamma=2.0
                
                bce_loss = torch.nn.functional.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction="none")
                pt = torch.exp(-bce_loss)
                loss_cls = alpha * (1 - pt) ** gamma * bce_loss
                loss_cls = loss_cls.mean()
                
            else:
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            
            total_loss = loss_box + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
            
            def to_cpu(tensor):
                return tensor.detach().cpu()

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                # "x": to_cpu(loss_x).item(),
                # "y": to_cpu(loss_y).item(),
                # "w": to_cpu(loss_w).item(),
                # "h": to_cpu(loss_h).item(),
                # "im": to_cpu(loss_im).item(),
                # "re": to_cpu(loss_re).item(),
                "box":to_cpu(loss_box).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
                "loss_box": loss_box
            }

            return output, total_loss, self.metrics

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes

        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor, 5 --> xywh & objectness_score
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2)
        self.stride = stride
        
        # x(bs,(num_classes + 5) * num_anchors,x,y) to x(bs,3,x,y, num_classes + 5)
        # x, y --> [(13,13),(26,26),(52,52)]
        bs, _, ny, nx = x.shape  
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if not self.training:
            x = x.view(bs, -1, self.no)

        return x

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list
                            if isinstance(layer[0], YOLOLayer) or isinstance(layer[0], YOLOLayer3D)]

        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

        self.has_3d_head = False
        self.identify_detection_head_indices()

    def identify_detection_head_indices(self):                
        
        ''' 
        REQUIRED For Detection head forward pass after adaptive fusion.
        contains tuple of [(i-1, i)] or [(i, i+1)]
            i-1 or i:
                preceeding conv that downsamples from 
                latent_space to filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
            i or i+1: 
                YOLO detection head 
        '''
        self.detection_head_indices = []
        self.detection_head_names = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if (
                module_def["type"] == "convolutional"
                and i + 1 < len(self.module_defs)
                and (self.module_defs[i + 1]["type"] == "yolo" or self.module_defs[i+1]['type'] == 'yolo_3d')
            ):
                
                if self.module_defs[i+1]['type'] == 'yolo_3d':
                    self.has_3d_head = True 
                    
                self.detection_head_indices.append(
                    (i, i+1)
                )               
                self.detection_head_names.append(
                    (self.module_defs[i], self.module_defs[i+1])
                )         

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:            
                x = module(x)                
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        # return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)        
        return yolo_outputs
    
    def forward_backbone(self, x):
        img_size = x.size(2)
        layer_outputs = []
        
        intermediate_features = []
        
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:            
                if (
                    module_def["type"] == "convolutional"
                    and i + 1 < len(self.module_defs)
                    # and self.module_defs[i + 1]["type"] == "yolo"
                    and (self.module_defs[i + 1]["type"] == "yolo" or self.module_defs[i+1]['type'] == 'yolo_3d')
                ):
                    intermediate_features.append(x)
                
                x = module(x)                                
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            layer_outputs.append(x)
    
        return intermediate_features
    
    def forward_detection_head(self, grid_features:Iterable[torch.tensor], image_size:int):
        
        assert len(grid_features) == len(self.detection_head_indices)
        
        for idx, (prev_conv_idx, det_head_idx) in enumerate(self.detection_head_indices):
            
            grid_features[idx] = self.module_list[prev_conv_idx](grid_features[idx])
            grid_features[idx] = self.module_list[det_head_idx][0](grid_features[idx], image_size)
            
        return grid_features
    
    def forward_detection_head_3d(self, grid_features:Iterable[torch.tensor], 
                                image_size:int, targets:torch.Tensor=None):
        
        assert len(grid_features) == len(self.detection_head_indices)

        
        total_loss = torch.zeros(1, device=grid_features[0].device)
        cls_loss = torch.zeros(1, device=grid_features[0].device)
        bbox_loss = torch.zeros(1, device=grid_features[0].device)
        conf_loss = torch.zeros(1, device=grid_features[0].device)
        
        metrics_all = {}
        
        for idx, (prev_conv_idx, det_head_idx) in enumerate(self.detection_head_indices):
            
            grid_features[idx] = self.module_list[prev_conv_idx](grid_features[idx])
            output, loss, metrics = self.module_list[det_head_idx][0](grid_features[idx], 
                                                                image_size, targets)
            grid_size = self.module_list[det_head_idx][0].grid_size
            grid_features[idx] = output
            
            total_loss += loss
            cls_loss += metrics['cls']
            bbox_loss += metrics['loss_box']
            conf_loss += metrics['conf']
            
            metrics_all[grid_size] = metrics

        return grid_features, total_loss, metrics_all, cls_loss, bbox_loss, conf_loss

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # Conv2d(5, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                # Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                
                ''' 
                Original input layer is 
                    torch.Size([32, 3, 3, 3])
                Modified input layer is 
                    torch.Size([32, 3 + lidar_channels, 3, 3])
                    
                Numel() from original = 864
                '''
                                
                if i == 0 and self.hyperparams['channels'] > 3:                    
                    numel_original = 864                                        
                    original_weights = torch.from_numpy(
                        weights[ptr:ptr+numel_original]).view(
                            [32, 3, 3, 3]
                        )
                    
                    repeating_channels = self.hyperparams['channels'] - 3
                    avg_weights = original_weights.mean(dim=1, keepdim=True)   
                    avg_weights = avg_weights.repeat(1, repeating_channels, 1, 1)                 
                    
                    new_weights = torch.cat((original_weights, avg_weights), dim=1)
                    conv_layer.weight.data.copy_(new_weights)                    

                    ptr += numel_original
                    continue
                
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def get_backbone_trainable_params(self, requires_grad:bool):
        
        indices = [idx for tup in self.detection_head_indices for idx in tup]
        backbone_params = []
        
        for i, module in enumerate(self.module_list):
            if i not in indices:
                for name, p in module.named_parameters():
                    backbone_params.append(p)
                    p.requires_grad = requires_grad
                    
        return backbone_params
        
    def get_detection_head_params(self, requires_grad:bool):
        
        detection_head_params = []
        
        for idx_pair in self.detection_head_indices:
            for module_idx in idx_pair:
                for p in self.module_list[module_idx].parameters():
                    p.requires_grad = requires_grad
                    detection_head_params.append(p)    
                    
        return detection_head_params       
