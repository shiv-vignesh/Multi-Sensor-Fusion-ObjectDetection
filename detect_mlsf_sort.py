import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from dataset_utils.sort_utils import Sort, trajectory_path, count_objects, dwell_time
from dataset_utils.enums import Enums
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiMLSFCollateFn

from test_mlsf import load_mlsf_yolo
from detect_mlsf import generate_video_from_frames
from model.yolo_utils import apply_sigmoid_activation, non_max_suppression, rescale_boxes

def draw_bbox_2d_opencv(tracked_objects, image_path, classes:list,
                        object_counter:dict, track_lifetime:dict, fps:int=1):
    
    image_arr = cv2.imread(image_path)
    facecolor_rgba = (0, 1, 0, 0.3)    
    tracked_ids = []
    
    for x1, y1, x2, y2, cls_pred, conf, _, _, tracked_id in tracked_objects:
        print(x1, y1, x2, y2, conf, cls_pred, tracked_id)
        label = f"{classes[int(cls_pred)]}: {tracked_id:.2f}"
        edge_color = Enums.KiTTi_class_colors[int(cls_pred)]   
        
        # Draw bounding box
        cv2.rectangle(image_arr, (int(x1), int(y1)), (int(x2), int(y2)), edge_color, 2)
        
        cv2.putText(image_arr, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge_color, 2)
        
        tracked_ids.append(tracked_id)
                
        # if tracked_id in trajectory:
        #     print(f'Tracked Id: {tracked_id} in Trajectory')
        #     points = trajectory[tracked_id]
        #     cv2.polylines(
        #         image_arr, [points], isClosed=False, color=(255, 0, 0), thickness=2
        #     )
    
    # === Create space for stats at the bottom ===
    stats_height = 100

    stats_background = np.zeros((stats_height, image_arr.shape[1], 3), dtype=np.uint8)
    
    # === Prepare text ===
    text_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_height = 20
    y_offset = 20
    
    if object_counter:
        for class_id, track_ids in object_counter.items():

            class_name = classes[int(class_id)]
            text = f"{class_name}: {len(track_ids)}"
            cv2.putText(stats_background, text, (10, y_offset), font, font_scale, text_color, 1)
            y_offset += line_height             
    
    if track_lifetime:
        for idx, (tid, tinfo) in enumerate(track_lifetime.items()):
            if tid in tracked_ids:
                continue
            start_f, end_f, cls_pred = tinfo['start'], tinfo['end'], tinfo['class']
            
            dwell_frames = (end_f - start_f)/10 #10 fps
            dwell_sec = dwell_frames / fps
            text = f"Class: {classes[int(cls_pred)]} ID {tid} Dwell Time: {dwell_sec:.1f}s"
            cv2.putText(stats_background, text, (250, 20 + idx * line_height), font, font_scale, text_color, 1)    
    
    image_arr = np.vstack((image_arr, stats_background))
    
    return image_arr

def create_dataloader_kitti(test_dataset_kwargs:dict, lidar_map_type:str="depth_map"):

    dataset = Kitti2DObjectDetectDataset(
        lidar_dir=test_dataset_kwargs['lidar_dir'],
        calibration_dir=test_dataset_kwargs['calibration_dir'],
        left_image_dir=test_dataset_kwargs['left_image_dir'],
        right_image_dir=test_dataset_kwargs['right_image_dir'],
        labels_dir=test_dataset_kwargs['labels_dir']
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=test_dataset_kwargs['batch_size'],
        collate_fn=KittiMLSFCollateFn(
            image_resize=test_dataset_kwargs['image_resize'],
            detection_head='yolo', 
            lidar_map_type=lidar_map_type
        ),
        shuffle=test_dataset_kwargs['shuffle']
    )
    
    return dataloader

def detect(mlsf, data_items, image_resize, original_shape):
    _, _, outputs = mlsf(
        data_items['images'],
        data_items['lidar_2d'] if mlsf.use_lidar_backbone else None,
        data_items['targets']
    )

    outputs = outputs['obj_2d']
    anchor_grids = [yolo_layer.anchor_grid for yolo_layer in mlsf.image_backbone.yolo_layers]

    outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
    outputs = non_max_suppression(outputs, conf_thres=0.15)
    
    # outputs[0] corresponds to the first (and only) image in the batch
    if outputs[0] is not None and len(outputs[0]): 
        detections = rescale_boxes(
            outputs[0], image_resize, original_shape # Pass image_resize directly if it's [h, w]
        )
    else:
        detections = torch.empty((0, 6)) # No detections found, return empty tensor with expected columns

    return detections

def track_objects(test_kwargs):
    
    # Initialize mot_tracker (SORT)
    mot_tracker = Sort(max_age=test_kwargs['tracker_kwargs']['max_age'], 
                    min_hits=test_kwargs['tracker_kwargs']['min_hits'], 
                    iou_threshold=test_kwargs['tracker_kwargs']['iou_threshold'])
    
    # Initialize mlsf (detector)
    mlsf = load_mlsf_yolo(test_kwargs['mlsf_yolo_kwargs'], test_kwargs['model_path'])
    
    # Initialize dataloader
    dataloader = create_dataloader_kitti(test_kwargs['kitti_sweeps_dataset_kwargs'], 
                                        lidar_map_type='bev_map')    
    
    class_names = list(Enums.KiTTi_label2Id.keys())
    image_resize = test_kwargs['kitti_sweeps_dataset_kwargs']['image_resize']
    save_dir = test_kwargs['save_dir']
    
    iter = tqdm(dataloader)
    tracker_history = {}    
    rgb_frames = []
    track_lifetime = defaultdict(lambda: {'start':None, 'end':None, 'class':None})
    
    for frame_idx, data_items in enumerate(iter):
        
        image_path = data_items['image_paths'][0]        
        original_shape = cv2.imread(image_path).shape[:2]
        active_track_ids = set()
        frame_idx += 1
        
        with torch.no_grad():
            frame_detections = detect(
                mlsf, data_items, image_resize[0], original_shape
            )
            if frame_detections is not None and frame_detections.shape[0] > 0:
                detections_for_sort = frame_detections.cpu().numpy()
            else:
                detections_for_sort = np.empty((0, 6))
            
            tracked_objects = mot_tracker.update(
                detections_for_sort
            )

            # trajectory = trajectory_path(tracked_objects, active_track_ids, 
            #                             tracker_history, frame_idx)
            
            object_counter = count_objects(tracked_objects)
            
            track_lifetime = dwell_time(tracked_objects, track_lifetime, frame_idx)

            image_arr = draw_bbox_2d_opencv(
                tracked_objects,
                data_items['image_paths'][0],
                class_names,
                object_counter, 
                track_lifetime
            )

            rgb_frames.append(image_arr)

    generate_video_from_frames(
        rgb_frames, 
        f'{save_dir}/mlsf_sort.mp4'
    )

if __name__ == "__main__":
        
    test_kwargs = {
        "mlsf_yolo_kwargs":{
            "image_cfg_file":"config/yolov3-kitti-608.cfg",
            "lidar_cfg_file":"config/yolov3-yolo_reduced_classes_3D.cfg",
            "image_channels":3, 
            "lidar_channels":3, 
            "image_backbone_device":"cuda:6",
            "lidar_backbone_device":"cuda:7",
            "adaptive_fusion_device":"cuda:8",
            "use_lidar_backbone":True,
            "apply_adaptive_fusion":True, 
            "fusion_type":"attention",
            "num_fusion_blocks":2, 
            "weighted_fusion":False,
            "model_seed":101
        },
        "tracker_kwargs":{
            "max_age":20, 
            "min_hits":3,
            "iou_threshold":0.4
        },
        "kitti_sweeps_dataset_kwargs":{
            "lidar_dir":"data/KiTTi/sweeps/2011_09_29/2011_09_29_drive_0071_sync/velodyne_points/data",
            "calibration_dir":None,
            "left_image_dir":"data/KiTTi/sweeps/2011_09_29/2011_09_29_drive_0071_sync/image_02/data",
            "right_image_dir":None,
            "labels_dir":None,
            "shuffle":False,
            "apply_augmentation":False, 
            "batch_size":1, #single frame at a time.
            "image_resize":[608, 608]
        }, 
        "model_path":"results/mlsf_yolov3_ckpts_608/MLSF-YOLOv3-608-RGB-BEV/best-model_obj_2d/best-model.pt",
        "save_dir":"data/KiTTi/sweeps/2011_09_29/2011_09_29_drive_0071_sync"
    }
    
    track_objects(test_kwargs)

    
    