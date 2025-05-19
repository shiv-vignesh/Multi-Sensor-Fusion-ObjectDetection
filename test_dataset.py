import torch, time
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn, KittiMLSFCollateFn, KittiMLSFMobilenet
from dataset_utils.enums import Enums
from dataset_utils.nuscenes_2d_objectDetect import NuScenesObjectDetectDataset, NuScenesMLSFCollateFn
from model.mlsf_utils import box_center_to_corner
# from trainer.loss import compute_loss
# from model.yolo import Darknet

from model.mlsf_yolo import MLSFYolo
from model.mlsf_mobilenet import MLSFMobilenet
# from model.mlsf_yolov8 import MLSFYolov8

if __name__ == "__main__":
    
    # yolo = Darknet(
    #     'config/yolov3-KiTTi.cfg'
    # )
    
    # image_resize = (416, 416)
    
    # mlsf = MLSF(config_path='config/yolov3-yolo_reduced_classes.cfg')
    # mlsf = MLSFYolov8(
    #     yolov8_weights_path='yolov8n.pt',
    #     fusion_type='attention',
    #     weighted_fusion=True,
    #     num_fusion_blocks=2,
    #     num_classes=len(Enums.KiTTi_label2Id)
    # )
    
    mlsf = MLSFYolo(
        image_config_path='config/yolov3-nuscenes-608.cfg',
        lidar_config_path='config/yolov3-nuscenes-608.cfg', 
        apply_adaptive_fusion=True, 
        use_lidar_backbone=True, 
        num_fusion_blocks=2, 
        fusion_type="attention",
        task_type=["obj_2d"], 
        weighted_fusion=False
    )
    
    dataset = NuScenesObjectDetectDataset(
        table_blob_paths=[
            'data/nuscenes/trainval03_blobs_US/tables.json', 
            'data/nuscenes/trainval04_blobs_US/tables.json'
        ], 
        root_dir=f'data/nuscenes'
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=NuScenesMLSFCollateFn(
            image_resize=(608, 608), 
            lidar_map_type='bev_map'
        ), 
        shuffle=True
    )
    
    for data_items in dataloader:
        for k,v in data_items.items():
            if torch.is_tensor(v):
                print(f'{k} {v.shape}')
        
        loss, loss_components, outputs = mlsf(
            data_items['images'],
            data_items['lidar_2d'],
            data_items['targets'], 
            data_items['targets_3d']
        )
        
        print(loss)
        
        exit(1)
    
    # mlsf = MLSFMobilenet(
    #     image_resize=(224, 224)
    # )
        
    # dataset = Kitti2DObjectDetectDataset(
    #     lidar_dir="data/KiTTi/training/velodyne",
    #     calibration_dir="data/KiTTi/training/calib",
    #     left_image_dir="data/KiTTi/training/image_2",
    #     labels_dir="data/KiTTi/training/label_2"

    # )

    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=2,
    #     collate_fn=KittiMLSFMobilenet(
    #         image_resize=(224, 224), 
    #         detection_head='yolo'
    #     ),
    #     shuffle=True
    # )        
    
    # for data_items in dataloader:
        
        # for k, v in data_items.items():
        #     if torch.is_tensor(v):
        #         print(f'{k} {v.shape}')
                
        # exit(1)
        
        # loss_2d, loss_2d_components, fused_features_list = mlsf(
        #     data_items['images'],
        #     data_items['lidar_2d'],
        #     data_items['targets']
        # )
        
        # print(loss_2d)
        # exit(1)
        
        # image_path = data_items['image_paths'][0]        
        # for depth in mlsf.detection_depths:
        #     anchor_boxes = mlsf.anchor_info[depth]['default_boxes']
        #     anchor_corners = box_center_to_corner(anchor_boxes)
            
        #     print(anchor_corners)
            
        # exit(1)
        # total_loss, loss_components, detections = mlsf(
        #     data_items['images'],
        #     data_items['lidar_depth_2d'],
        #     data_items['targets'],
        # )
        
        # outputs = mlsf(
        #     data_items['images'],
        #     data_items['lidar_depth_2d'],
        #     data_items['targets'],
        # )        
        
        # print(outputs)

        # exit(1)        