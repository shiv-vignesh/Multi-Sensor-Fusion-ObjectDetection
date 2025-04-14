import json, os
import torch 

from dataset_utils.enums import Enums
from model.mlsf_yolo import MLSFYolo
from model.mlsf_yolov8 import MLSFYolov8
from model.mlsf_mobilenet import MLSF
from trainer.trainer_mlsf_yolo import MLSFTrainerYolo
from trainer.trainer_mlsf_ssd import MLSFTrainerSSD

def load_torch_mismatch_weights(model:MLSFYolo, pth_file:str, strict_loading:bool=True):
    
    weights_dict = torch.load(pth_file, map_location='cpu')        
    model_dict = model.state_dict()
        
    for idx, (model_layer, weight_layer) in enumerate(zip(model_dict, weights_dict)):

        layer_shape = model_dict[model_layer].shape
        weight_shape = weights_dict[weight_layer].shape
        
        if layer_shape == weight_shape:
            print(f'Match at {model_layer}: Weights Shape: {weight_shape} - Layer Shape: {layer_shape}')
            model_dict[model_layer] = weights_dict[weight_layer]
            
        else:
            if strict_loading:
                exit(1)
                
            print(f'Mismatch at {model_layer}: Weights Shape: {weight_shape} - Layer Shape: {layer_shape}')
            if 'weight' in model_layer:
                model_dict[model_layer] = torch.nn.init.normal_(
                    model_dict[model_layer], 0.0, 0.02
                )
                
            elif 'bias' in model_layer:
                model_dict[model_layer] = torch.nn.init.constant_(
                    model_dict[model_layer], 0.0
                )
            
            
    return model.load_state_dict(model_dict)

def create_mlsf_yolo(model_kwargs:dict, model_type:str, weights_path:str=None, image_resize:tuple=(416, 416)):
    
    image_backbone_device = torch.device(model_kwargs['image_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    lidar_backbone_device = torch.device(model_kwargs['lidar_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    adaptive_fusion_device = torch.device(model_kwargs['adaptive_fusion_device']) if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_kwargs['model_seed'])
        torch.cuda.manual_seed_all(model_kwargs['model_seed'])
        
    else:
        torch.manual_seed(model_kwargs['model_seed'])

    if model_type == "mlsf_yolov3":    
        mlsf = MLSFYolo(
            image_config_path=model_kwargs['image_cfg_file'], 
            lidar_config_path=model_kwargs['lidar_cfg_file'], 
            image_backbone_device=image_backbone_device, 
            lidar_backbone_device=lidar_backbone_device,
            adaptive_fusion_device=adaptive_fusion_device, 
            apply_adaptive_fusion=model_kwargs['apply_adaptive_fusion'], 
            num_fusion_blocks=model_kwargs['num_fusion_blocks'], 
            fusion_type=model_kwargs['fusion_type'], 
            weighted_fusion=model_kwargs['weighted_fusion'],
            task_type=model_kwargs['task_type']
        )        
        
        if weights_path:
            if weights_path.endswith(".pt") or weights_path.endswith(".pth"):
                # Load checkpoint weights                
                mlsf.to(
                    torch.device('cpu')
                )

                try:
                    mlsf.load_state_dict(
                        torch.load(weights_path)
                    )
                    
                except:
                    
                    print(f'Loading with Mismatch for MLSF Image Backbone: {mlsf.image_backbone.__class__.__name__}')
                    load_torch_mismatch_weights(mlsf.image_backbone.cpu(), weights_path, strict_loading=False)
                    print(f'Loading with Mismatch for MLSF LiDAR Backbone: {mlsf.lidar_backbone.__class__.__name__}')
                    load_torch_mismatch_weights(mlsf.lidar_backbone.cpu(), weights_path, strict_loading=False)
                    print(f'Done Loading')
                    
                
                mlsf.image_backbone.to(mlsf.image_backbone_device)
                mlsf.lidar_backbone.to(mlsf.lidar_backbone_device)
                
                if mlsf.apply_adaptive_fusion:
                    mlsf.adaptive_fusion_module.to(mlsf.adaptive_fusion_device)

            else:
                # Load darknet weights
                print(f'Loading Darknet weights')
                mlsf.image_backbone.load_darknet_weights(weights_path)    
                if mlsf.use_lidar_backbone:
                    mlsf.lidar_backbone.load_darknet_weights(weights_path)
                    
    if model_type == "mlsf_yolov8":
        """
        TODO, add load model ckpt
        """
        mlsf = MLSFYolov8(
            yolov8_weights_path=model_kwargs['yolov8_weights_path'],
            fusion_type=model_kwargs['fusion_type'],
            weighted_fusion=model_kwargs['weighted_fusion'],
            num_fusion_blocks=model_kwargs['num_fusion_blocks'],
            num_classes=len(Enums.KiTTi_label2Id), 
            image_backbone_device=image_backbone_device, 
            lidar_backbone_device=lidar_backbone_device,
            adaptive_fusion_device=adaptive_fusion_device, 
            image_resize=image_resize
        )
        
        if weights_path:
            if weights_path.endswith(".pt"):
                # Load checkpoint weights
                
                mlsf.to(
                    torch.device('cpu')
                )
                
                try:                
                    mlsf.load_state_dict(
                        torch.load(weights_path)
                    )
                except:
                    print(f'Loading with Mismatch')
                    load_torch_mismatch_weights(mlsf.cpu(), weights_path)
                    print(f'Done Loading')    
                    
                    exit(1)
                
                mlsf.image_backbone.to(mlsf.image_backbone_device)
                mlsf.lidar_backbone.to(mlsf.lidar_backbone_device)
                mlsf.adaptive_fusion_module.to(mlsf.adaptive_fusion_device)
                mlsf.detection_heads.to(mlsf.adaptive_fusion_device)
    
    return mlsf

def create_mlsf_ssd(model_kwargs:dict, weights_path:str):

    # mlsf = MLSF(
    #     config_path=model_kwargs['cfg_file'],
    #     image_channels=model_kwargs['image_channels'],
    #     lidar_channels=model_kwargs['lidar_channels'],
    #     fm_size=model_kwargs['feature_map_size'],
    #     image_size=tuple(model_kwargs['image_resize']), 
    #     image_backbone_device=image_backbone_device, 
    #     lidar_backbone_device=lidar_backbone_device, 
    #     adaptive_fusion_device=adaptive_fusion_device, 
    #     detection_depths=model_kwargs['detection_depths']
    # )

    image_backbone_device = torch.device(model_kwargs['image_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    lidar_backbone_device = torch.device(model_kwargs['lidar_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    adaptive_fusion_device = torch.device(model_kwargs['adaptive_fusion_device']) if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_kwargs['model_seed'])
        torch.cuda.manual_seed_all(model_kwargs['model_seed'])
        
    else:
        torch.manual_seed(model_kwargs['model_seed'])
    
    mlsf = MLSF(
        image_size=model_kwargs['image_resize'],
        detection_depths=model_kwargs['detection_depths'],
        apply_adaptive_fusion=model_kwargs['apply_adaptive_fusion'],
        use_lidar_backbone=model_kwargs['use_lidar_backbone'],
        image_backbone_device=image_backbone_device, 
        lidar_backbone_device=lidar_backbone_device, 
        adaptive_fusion_device=adaptive_fusion_device,         
    )
    
    if os.path.exists(weights_path):
        mlsf.load_state_dict(
            torch.load(weights_path)
        )
        
        mlsf.image_backbone.to(mlsf.image_backbone_device)
        
        if mlsf.use_lidar_backbone:
            mlsf.lidar_backbone.to(mlsf.lidar_backbone_device) 
        
        if mlsf.apply_adaptive_fusion:
            mlsf.adaptive_fusion_module.to(mlsf.adaptive_fusion_device)
        
        mlsf.ssd_detection_heads.to(mlsf.adaptive_fusion_device)
    
    return mlsf
        
if __name__ == "__main__":
    
    trainer_config = json.load(open('config/mlsf_trainer.json'))    
    # darknet53_path = ''
    # weights_pth_path = "training_logs/pretrained_darknet53_rgb_Lidar/yolo_weights_59.pth"
    
    if trainer_config['model_kwargs']['model_type'] == 'mlsf_yolov3' or trainer_config['model_kwargs']['model_type'] == 'mlsf_yolov8':   
        darknet53_path = 'darknet53.conv.74'
        # model_path = 'MLSF-YOLOv8-nano-Attention/ckpt_2/ckpt-model.pt'
        # model_path = 'MLSF-YOLOv3-3D-Bev-(LiDARBackbone-only)-3/ckpt_14/ckpt-model.pt'
        # model_path = 'results/MLSF-YOLO-Attention-FocalLoss/best-model/best-model.pt'
        model_path = "MLSF-YOLOv3-JointTraining/best-model_obj_2d/best-model.pt"
        
        # model_path = 'yolov3_ckpt_epoch-298.pth'

        mlsf = create_mlsf_yolo(
            trainer_config['model_kwargs']['mlsf_yolo_kwargs'], 
            trainer_config['model_kwargs']['model_type'],
            model_path, 
            image_resize=tuple(trainer_config['dataset_kwargs']['image_resize'])
        )

        trainer = MLSFTrainerYolo(
            mlsf=mlsf, 
            dataset_kwargs=trainer_config['dataset_kwargs'],
            optimizer_kwargs=trainer_config['optimizer_kwargs'],
            trainer_kwargs=trainer_config['trainer_kwargs'],
            lr_scheduler_kwargs=trainer_config['lr_scheduler_kwargs']
        )
        
        trainer.train()
        
    elif trainer_config['model_kwargs']['model_type'] == 'mlsf_ssd':   
        
        weights_path = 'MLSF-SSD-Training/ckpt_14/mlsf_ssd.pth'
        mlsf = create_mlsf_ssd(
            trainer_config['model_kwargs']['mlsf_ssd_kwargs'], weights_path
        )                
        
        trainer = MLSFTrainerSSD(
            mlsf=mlsf, 
            dataset_kwargs=trainer_config['dataset_kwargs'],
            optimizer_kwargs=trainer_config['optimizer_kwargs'],
            trainer_kwargs=trainer_config['trainer_kwargs'],
            lr_scheduler_kwargs=trainer_config['lr_scheduler_kwargs']            
        )
        
        trainer.train()