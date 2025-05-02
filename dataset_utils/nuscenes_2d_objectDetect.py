import os
import json
from typing import List, Dict, Iterable

import cv2 
import albumentations
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from dataset_utils.enums import Enums

class NuScenesObjectDetectDataset(Dataset):    
    
    def __init__(self, table_blob_paths:list, root_dir:str):
        
        self.sample_tokens = []
        self.tables = {}
        self.table_blob_paths = table_blob_paths
        
        self.root_dir = root_dir
                
        for blob_table_path in table_blob_paths:
            self.parse_blob_table(blob_table_path)
    
    def parse_blob_table(self, blob_table_path:str):
        
        table = json.load(open(blob_table_path))
        self.sample_tokens.extend([sample_token for sample_token in table])
        self.tables.update(table)

    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        
        sample_token = self.sample_tokens[idx]
        lidar_top_fp = self.tables[sample_token]['lidar_top_fp']
        cam_front_fp = self.tables[sample_token]['cam_front_fp']
        labels_2d_cam_front = self.tables[sample_token]['labels_2d_cam_front']
        
        # splitting because 
        # ../data/nuscenes/trainval04_blobs_US/samples/CAM_FRONT/filename.jpg
        lidar_top_fp = f"{self.root_dir}/{'/'.join(lidar_top_fp.split('/')[3:])}"
        cam_front_fp = f"{self.root_dir}/{'/'.join(cam_front_fp.split('/')[3:])}"
        
        return {
            'sample_token':sample_token, 
            'lidar_top_fp':lidar_top_fp, 
            'cam_front_fp':cam_front_fp, 
            'labels_2d_cam_front':labels_2d_cam_front
        }
        
class NuScenesMLSFCollateFn(object):
    
    def __init__(self, image_resize:list, original_size:tuple=(900, 1600), 
                lidar_map_type:str='depth_map', transformation=None, 
                apply_augmentation=False):
        
        self.image_resize = image_resize
        self.original_size = original_size
        
        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]
        
        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]
                
        self.apply_augmentation = apply_augmentation
        self.lidar_map_type = lidar_map_type

        if transformation is None:
            self.transformation = albumentations.Compose(
                # [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)],
                [albumentations.LongestMaxSize(max_size=max(self.image_resize)),
                albumentations.PadIfNeeded(
                    min_height=self.resized_height,
                    min_width=self.resized_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ), 
                albumentations.CenterCrop(height=self.resized_height, width=self.resized_width, always_apply=True)
                ],                
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )

            self.image_only_transformation = albumentations.Compose(
                [albumentations.Resize(height=self.resized_height, width=self.resized_width, always_apply=True)]
            )

        # Front side (of vehicle) Point Cloud boundary for BEV
        self.boundary = {
            "minX": -25,
            "maxX": 25,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.5,
            "maxZ": 1.5
        }
        
        self.bev_res = 0.1

    def transform_sample(self, image:np.array, label_bboxes:np.array=None, class_labels:np.array=None):                        

        if label_bboxes is not None and class_labels is not None:        
            transformed_dict = self.transformation(
                image=image, bboxes=label_bboxes, class_labels=class_labels
            )
        else:
            transformed_dict = self.image_only_transformation(
                image=image
            )
        return transformed_dict

    def lidar_to_depth_map(self, lidar_point_cloud:np.array):
        
        # Euclidean distance of the point from lidar sensor mounted on car
        # aka depth. (Equation 1)        
        
        #Azimuthal Angle Mapping (Equation 2)
        # x_depth = tan^-1 (y_3d/x_3d)

        #Elevation Mapping (Equation 3)
        #Capture vertical component; y_depth = Cos^-1 (z_3d/d_3d)        

        # Extract x, y, z coordinates
        x_3d = lidar_point_cloud[:, 0]
        y_3d = lidar_point_cloud[:, 1]
        z_3d = lidar_point_cloud[:, 2]

        # Compute depth values (Euclidean distance)
        depth_3d = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)

        # Select front 90 degrees (-45 to 45 degrees)
        azimuth = np.degrees(np.arctan2(y_3d, x_3d))
        mask = (azimuth > -45) & (azimuth < 45)
        x_3d, y_3d, z_3d, depth_3d = x_3d[mask], y_3d[mask], z_3d[mask], depth_3d[mask]

        # Compute azimuth and elevation angles
        x_depth_map = np.arctan2(y_3d, x_3d)  # Horizontal mapping
        y_depth_map = np.arccos(np.clip(z_3d / depth_3d, -1.0, 1.0))  # Vertical mapping
        
        # Normalize values to image dimensions
        x_min, x_max = x_depth_map.min(), x_depth_map.max()
        y_min, y_max = y_depth_map.min(), y_depth_map.max()
        
        x_img = ((x_depth_map - x_min) / (x_max - x_min) * self.original_width).astype(int)
        y_img = ((y_depth_map - y_min) / (y_max - y_min) * self.original_height).astype(int)
        
        # Ensure indices are within valid range
        x_img = np.clip(x_img, 0, self.original_width - 1)
        y_img = np.clip(y_img, 0, self.original_height - 1)
        
        # Create an empty image
        depth_map = np.zeros((self.original_height, self.original_width))
        
        # Fill in the depth values (color encoding)
        for i in range(len(depth_3d)):
            depth_map[y_img[i], x_img[i]] = depth_3d[i]
        
        # Normalize and apply colormap
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)                
        depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        depth_map_colored = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)

        return depth_map_colored
        
    def generate_bev_map(self, lidar_point_cloud: np.ndarray):
        """
        Generates BEV map from nuScenes LIDAR_TOP point cloud.
        
        Returns:
            RGB_Map: np.ndarray of shape (3, H, W) with intensity (R), height (G), and density (B)
        """

        bev_width = int((self.boundary['maxX'] - self.boundary['minX']) / self.bev_res)
        bev_height = int((self.boundary['maxY'] - self.boundary['minY']) / self.bev_res)

        # Initialize 3-channel BEV map: Height (R), Intensity (G), Density (B)
        bev_map = np.zeros((3, bev_height, bev_width), dtype=np.float32)
        density_map = np.zeros((bev_height, bev_width), dtype=np.float32)

        # Filter points within the specified boundaries
        mask = (
            (lidar_point_cloud[:, 0] >= self.boundary['minX']) & (lidar_point_cloud[:, 0] <= self.boundary['maxX']) &
            (lidar_point_cloud[:, 1] >= self.boundary['minY']) & (lidar_point_cloud[:, 1] <= self.boundary['maxY']) &
            (lidar_point_cloud[:, 2] >= self.boundary['minZ']) & (lidar_point_cloud[:, 2] <= self.boundary['maxZ'])
        )
        points_filtered = lidar_point_cloud[mask]

        # Translate LiDAR points to BEV coordinates (x, y)
        x_img = ((points_filtered[:, 0] - self.boundary['minX']) / self.bev_res).astype(np.int32)
        y_img = ((points_filtered[:, 1] - self.boundary['minY']) / self.bev_res).astype(np.int32)

        # Clip to valid range
        x_img = np.clip(x_img, 0, bev_width - 1)
        y_img = np.clip(y_img, 0, bev_height - 1)

        # Height map: Assign max height per pixel
        for i in range(points_filtered.shape[0]):
            x = x_img[i]
            y = y_img[i]
            z = points_filtered[i, 2]
            intensity = points_filtered[i, 3]
            
            # Height map (Red channel)
            bev_map[0, y, x] = max(bev_map[0, y, x], z)
            
            # Intensity map (Green channel)
            bev_map[1, y, x] = max(bev_map[1, y, x], intensity)

            # Density map calculation
            density_map[y, x] += 1  # Count points per pixel

        # Normalize the height map to range [0, 1] (for Red channel)
        bev_map[0] = (bev_map[0] - self.boundary['minZ']) / (self.boundary['maxZ'] - self.boundary['minZ'])

        # Normalize intensity map (for Green channel)
        bev_map[1] = np.clip(bev_map[1], 0, 1)

        # Normalize the density map using logarithmic scaling (for Blue channel)
        density_map = np.log1p(density_map) / np.log(64)  # Log normalization (log(N + 1) / log(64))
        density_map = np.clip(density_map, 0, 1)  # Clipping to [0, 1]

        # Assign density map to Blue channel (Blue -> Density)
        bev_map[2] = density_map

        # Stack the channels (Height -> Red, Intensity -> Green, Density -> Blue)
        bev_rgb = np.stack([
            (bev_map[0] * 255).astype(np.uint8),  # Height -> Red
            (bev_map[1] * 255).astype(np.uint8),  # Intensity -> Green
            (bev_map[2] * 255).astype(np.uint8),  # Density -> Blue
        ], axis=-1)

        # Resize the output BEV image to 608x608
        bev_resized = cv2.resize(bev_rgb, (self.resized_width, self.resized_height), interpolation=cv2.INTER_NEAREST)

        return bev_resized
            
    def preprocess_labels(self, labels_2d_cam_front:Iterable[dict]):

        class_labels = []
        bboxes_2d = []
        
        for label_data in labels_2d_cam_front:

            attribute = label_data['category_name']
            if attribute in Enums.NUSCENES_TO_GENERAL_CLASSES:
                
                if Enums.NUSCENES_TO_GENERAL_CLASSES[attribute] not in Enums.nuscenes_label2Id:
                    continue

                bbox_corners = label_data['bbox_corners']

                left = float(bbox_corners[0])  # left
                top = float(bbox_corners[1])  # top
                right = float(bbox_corners[2])  # right
                bottom = float(bbox_corners[3])  # bottom
                
                class_label = Enums.NUSCENES_TO_GENERAL_CLASSES[attribute]
                class_id = Enums.nuscenes_label2Id[class_label]
                
                class_labels.append(class_id)
                bboxes_2d.append([left, top, right, bottom])

        return class_labels, bboxes_2d
    
    def prepare_targets_2d(self, batch_idx:int, class_labels:list, class_bboxes:list, letter_box:bool=True):

        targets = []
        for bbox, class_id in zip(class_bboxes, class_labels):        
            left, top, right, bottom = bbox

            x_center = (left + right) / 2 / self.image_resize[1]
            y_center = (top + bottom) / 2 / self.image_resize[0]
            width = (right - left) / self.image_resize[1]
            height = (bottom - top) / self.image_resize[0]

            targets.append([
                batch_idx, class_id, x_center, y_center, width, height
            ])

        return targets    

    def __call__(self, batch_data:List[Dict]):
        
        batch_data_items = {
            'images':[],
            "image_paths":[],
            'lidar_2d':[],
            "targets": [],
            "targets_3d":[]
        }
        
        for idx, data in enumerate(batch_data):
            sample_token = data['sample_token']
            lidar_top_fp = data['lidar_top_fp']
            cam_front_fp = data['cam_front_fp']
            labels_2d_cam_front = data['labels_2d_cam_front']

            if os.path.exists(lidar_top_fp) and os.path.exists(cam_front_fp):
                lidar_point_cloud = np.fromfile(lidar_top_fp, dtype=np.float32).reshape(-1, 5)
                cam_front = cv2.imread(cam_front_fp)
                batch_data_items['image_paths'].append(cam_front_fp)

            else:
                print(f'Camera Front: {cam_front_fp} Does not Exists!')
                print(f'Lidar Top: {lidar_top_fp} Does not Exists!')
                exit(1)

            if self.lidar_map_type == 'depth_map':
                lidar_map = self.lidar_to_depth_map(lidar_point_cloud[:, :4])
                
            elif self.lidar_map_type == 'bev_map':
                lidar_map = self.generate_bev_map(lidar_point_cloud)

            if labels_2d_cam_front:
                class_labels, label_bboxes_2d = self.preprocess_labels(labels_2d_cam_front)
                
                transformed_dict = self.transform_sample(
                    cam_front, label_bboxes_2d, class_labels
                )
                
                targets = self.prepare_targets_2d(
                    idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                ) #prepared targets within (0 to 1) normalized, with image_resize
                
                batch_data_items['targets'].append(
                    torch.tensor(targets, dtype=torch.float32)
                )
                
                cam_front_image = transformed_dict['image']
                if self.lidar_map_type == 'depth_map':
                    lidar_map = self.transform_sample(lidar_map)['image']

            else:
                cam_front_image = self.transform_sample(
                    cam_front
                )['image']

                lidar_map = self.transform_sample(lidar_map)['image']

            cam_front_image = torch.from_numpy(cam_front_image).permute((2, 0, 1)) #(nc, h, w)

            if self.lidar_map_type == 'depth_map':
                lidar_map_tensor = torch.from_numpy(lidar_map).permute((2, 0, 1)) #(nc, h, w)
            elif self.lidar_map_type == 'bev_map':
                lidar_map_tensor = torch.from_numpy(lidar_map).permute((2, 0, 1)) #(nc, h, w)

            batch_data_items['images'].append(cam_front_image)
            batch_data_items['lidar_2d'].append(lidar_map_tensor)
            
        batch_data_items['images'] = torch.stack(
            batch_data_items['images'], dim=0
        ).float()        
        
        batch_data_items['lidar_2d'] = torch.stack(
            batch_data_items['lidar_2d'], dim=0
        ).float()     

        if batch_data_items['targets']:
            batch_data_items['targets'] = torch.concat(
                batch_data_items['targets'], dim=0
            )

        return batch_data_items            

if __name__ == "__main__":
    
    dataset = NuScenesObjectDetectDataset(
        [
            '../data/nuscenes/trainval03_blobs_US/tables.json', 
            '../data/nuscenes/trainval04_blobs_US/tables.json'
        ]
    )
    
    
    
