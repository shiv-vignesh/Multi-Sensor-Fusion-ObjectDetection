import os, random, json
import torch
import cv2

from collections import defaultdict

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from dataset_utils.enums import Enums

from test_mlsf import load_mlsf_yolo, create_dataloader_kitti
from model.yolo_utils import apply_sigmoid_activation, non_max_suppression, rescale_boxes

def generate_video_from_frames(frames, output_path, fps=10):
    
    if not frames:
        print("No frames to render.")
        return
    
    # Read first frame to get dimensions
    height, width, _ = frames[0].shape

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")

def draw_bbox_2d_opencv(detections, image_path, 
                    resized_image_size:int, classes:list, 
                    stats:dict):
    
    image_arr = cv2.imread(image_path)    
    detections = rescale_boxes(detections, resized_image_size, image_arr.shape[:2])
    facecolor_rgba = (0, 1, 0, 0.3)

    for x1, y1, x2, y2, conf, cls_pred in detections:
        label = f"{classes[int(cls_pred)]}: {conf:.2f}"
        edge_color = Enums.KiTTi_class_colors[int(cls_pred)]        

        # Draw bounding box
        cv2.rectangle(image_arr, (int(x1), int(y1)), (int(x2), int(y2)), edge_color, 2)
        
        cv2.putText(image_arr, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, edge_color, 2)        
        cls_str = classes[int(cls_pred)]
        stats[cls_str] += 1

    return image_arr, stats

def draw_bbox_2d(detections, image_path, resized_image_size:int, classes:list,
                save_dir:str):

    image_arr = cv2.imread(image_path)
    image_arr = image_arr[:, :, ::-1]
    
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image_arr)      
    
    detections = rescale_boxes(detections, resized_image_size, image_arr.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    
    for x1, y1, x2, y2, conf, cls_pred in detections:

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle(
            (x1, y1), box_w, box_h, 
            linewidth=0.8, 
            edgecolor=color, 
            facecolor=(*mcolors.to_rgba(color)[:3], 0.3))

        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            fontsize=6,  # Small but readable
            verticalalignment="top",
            bbox={"color": color, "alpha": 0.6, "pad": 1, "linewidth": 0}  # Semi-transparent background
        )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    
    if not os.path.exists(f'{save_dir}/frames'):
        os.makedirs(f'{save_dir}/frames')

    frame_idx = image_path.split('/')[-1].split('.')[0]
    
    frame_path = os.path.join(f'{save_dir}/frames', f"frame_{frame_idx}.png")
    plt.savefig(frame_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

def detect(test_kwargs:dict):
    
    mlsf = load_mlsf_yolo(test_kwargs['mlsf_yolo_kwargs'], test_kwargs['model_path'])
    
    dataloader = create_dataloader_kitti(test_kwargs['kitti_sweeps_dataset_kwargs'], lidar_map_type='bev_map')
    image_resize = test_kwargs['kitti_sweeps_dataset_kwargs']['image_resize']
    class_names = list(Enums.KiTTi_label2Id.keys())
    
    save_dir = test_kwargs['save_dir']
    
    stats = defaultdict(int)
    detections_dict = defaultdict(lambda : defaultdict())
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    iter = tqdm(dataloader)
    for batch_idx, data_items in enumerate(iter):
        with torch.no_grad():
            loss, _, outputs = mlsf(
                data_items['images'],
                data_items['lidar_2d'] if mlsf.use_lidar_backbone else None,
                data_items['targets']
            )
            
            outputs = outputs['obj_2d']
            anchor_grids = [yolo_layer.anchor_grid for yolo_layer in mlsf.image_backbone.yolo_layers]
            
            outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
            outputs = non_max_suppression(outputs, conf_thres=0.15)

            if mlsf.use_lidar_backbone:
                data_items['lidar_2d'] = data_items['lidar_2d'][0].cpu().numpy()
                lidar_image = np.zeros((data_items['lidar_2d'].shape[1], data_items['lidar_2d'].shape[2], 3), dtype=np.uint8)
                for i in range(3):
                    lidar_image[:, :, i] = np.clip(data_items['lidar_2d'][i] * 255, 0, 255).astype(np.uint8)
            
            else:
                lidar_image = None

            image_arr, stats = draw_bbox_2d_opencv(
                outputs[0],
                data_items['image_paths'][0],
                image_resize[0], 
                class_names, stats
            )
            
            detections_dict[batch_idx]['detected_frame'] = image_arr
            detections_dict[batch_idx]['lidar_map'] = lidar_image
            detections_dict[batch_idx]['frame'] = cv2.imread(data_items['image_paths'][0])
            
    rgb_frames = [detections_dict[batch_idx]['frame'] for batch_idx in detections_dict]
    lidar_frames = [detections_dict[batch_idx]['lidar_map'] for batch_idx in detections_dict]
    detected_frames = [detections_dict[batch_idx]['detected_frame'] for batch_idx in detections_dict]
        
    generate_video_from_frames(
        rgb_frames,
        f'{save_dir}/rgb_cam.mp4'
    )
    
    generate_video_from_frames(
        lidar_frames,
        f'{save_dir}/lidar_maps.mp4'
    )
    
    generate_video_from_frames(
        detected_frames,
        f'{save_dir}/detection.mp4'
    )
    
    with open(f'{save_dir}/stats.json','w+') as f:
        json.dump(stats, f)
    
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

    detect(
        test_kwargs
    )