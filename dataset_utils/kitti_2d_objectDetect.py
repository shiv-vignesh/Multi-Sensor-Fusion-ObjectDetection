import os, open3d
import cv2 
import albumentations
import numpy as np
from time import time
import random

from typing import List, Dict, Iterable

import torch
import torchvision
from torch.utils.data import Dataset
# from imgaug import augmenters as iaa

from .enums import Enums

#monkey patch for iaa augmentation
np.bool = bool

def draw_bev_with_boxes(bev_map, targets, boundary, grid_x_res, grid_y_res, class_colors=None):
    """Draw the bounding boxes on top of the BEV map, using the class_id for coloring."""
    
    bev_image = np.zeros((bev_map.shape[1], bev_map.shape[2], 3), dtype=np.uint8)
    class_colors = {
        0: (255, 0, 0),  # Red (class 0)
        1: (0, 255, 0),  # Green (class 1)
        2: (0, 0, 255)   # Blue (class 2)
    }

    # Map intensity, height, and density to color channels (or just use one channel if needed)
    for i in range(3):
        bev_image[:, :, i] = np.clip(bev_map[i] * 255, 0, 255).astype(np.uint8)

    for target in targets:
        # Extract target info
        batch_idx, class_id, y_l, x_l, w, l, sin_yaw, cos_yaw = target

        # Convert (x, y) from normalized BEV coordinates to pixel coords
        px = int(x_l * bev_map.shape[2])  # x_l normalized to width
        py = int(y_l * bev_map.shape[1])  # y_l normalized to height

        # Size in BEV pixels (ensure they are large enough for visibility)
        pw = int(w * bev_map.shape[2])  # Width in pixels
        pl = int(l * bev_map.shape[1])  # Length in pixels

        # Calculate the yaw from sin/cos
        yaw = np.arctan2(sin_yaw, cos_yaw)

        # Create a rotated rectangle with the given center (px, py), size (pw, pl), and yaw
        rect = ((py, px), (pw, pl), np.degrees(yaw))  # Rotate in degrees
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        text_position = tuple(box_pts[0])  

        # If class_colors is provided, use it to color the boxes by class_id
        color = class_colors.get(class_id, (0, 255, 0)) if class_colors else (0, 255, 0)  # Default: green
        cv2.drawContours(bev_image, [box_pts], 0, color, 2)  # Draw the contour (bounding box)
        cv2.putText(bev_image, Enums.KiTTi_Id2label[class_id], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return bev_image

class LidarPreprocessorUtils:
    
    def transform_lidar_points(self, point_cloud_array:np.array, calibration_dict:dict):

        # ignoring the last_dim as it corresponds to intensity.
        if point_cloud_array.shape[-1] > 3:
            point_cloud_array = point_cloud_array[:, :3] #(N, 3)
        
        #reshaping rectification matrix from (12,) to (3,3)
        r0_rect = calibration_dict['R0_rect'].reshape(3,3) #(3,3)
        r0_rect_homo = np.vstack([r0_rect, [0, 0, 0]]) #(4,3)
        r0_rect_homo = np.column_stack([r0_rect_homo, [0, 0, 0, 1]]) #(4,4)
        
        # reshaping projection_matrix from (12,) to (3,4)
        proj_mat = calibration_dict['P2'].reshape(3,4) 
        
        # reshaping Tr_velo_to_cam from (12,) to (3,4)
        v2c = calibration_dict['Tr_velo_to_cam'].reshape(3,4)
        v2c = np.vstack(
            (v2c, [0, 0, 0, 1])
        ) #(4,4)    
        
        p_r0 = np.dot(proj_mat, r0_rect_homo) # (3, 4)
        p_r0_rt = np.dot(p_r0, v2c) #(3, 4)
        
        point_cloud_array = np.column_stack(
            [point_cloud_array, np.ones((point_cloud_array.shape[0], 1))]
        ) # (N, 4)
        
        #(3, 4) dot (4, N) ---> (3, N) ---> (N, 3)
        p_r0_rt_x = np.dot(
            p_r0_rt, point_cloud_array.T
        ).T 
        
        # # The transformed coordinates are for LIDAR (u, v, z) to (u', v', z') in Image. Normalize by depth (z')
        p_r0_rt_x[:, 0] /= p_r0_rt_x[:, -1]
        p_r0_rt_x[:, 1] /= p_r0_rt_x[:, -1]

        depth = p_r0_rt_x[:, -1]        
        negative_depth_mask = depth < 0
        
        return p_r0_rt_x[:, :2], p_r0_rt_x[:, -1], negative_depth_mask
        
        # proj_points_2d = p_r0_rt_x[:, :2]
        # depth = p_r0_rt_x[:, -1]
        
        # nonzero_depth_mask = depth > 0
        # # proj_points_2d = proj_points[nonzero_depth_mask]
        
        # depth[nonzero_depth_mask] = 1e-5
        # # valid_depth = depth > 1e-5 
        
        # proj_points_2d[:, 0] /= depth
        # proj_points_2d[:, 1] /= depth
        
        # return proj_points_2d[:, :2], depth
    
    def voxel_downsampling_torch(self, lidar_point_cloud:torch.tensor, 
                                 voxel_size = 0.2,  # Voxel size
                                 num_points:int=50000):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        lidar_point_cloud = lidar_point_cloud.to(device)
        voxel_indices = torch.floor(lidar_point_cloud[:, :3] / voxel_size).int()  # (N, 3)

        # Step 2: Group points by voxel
        unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)

        # Step 3: Aggregate points within each voxel (centroid or random)
        num_voxels = unique_voxels.size(0)
        aggregated_points = []
        for voxel_idx in range(num_voxels):
            mask = inverse_indices == voxel_idx
            voxel_points = lidar_point_cloud[mask]
            centroid = voxel_points.mean(dim=0)  # Compute centroid
            aggregated_points.append(centroid)

        downsampled_points = torch.stack(aggregated_points, dim=0)  # (M, D), where M = num_voxels

        # Step 4: Subsample or pad to fixed number of points if required
        if num_points:
            num_current_points = downsampled_points.size(0)
            if num_current_points > num_points:
                indices = torch.randperm(num_current_points)[:num_points]
                downsampled_points = downsampled_points[indices]
            elif num_current_points < num_points:
                padding = torch.zeros((num_points - num_current_points, lidar_point_cloud.size(1)), device=lidar_point_cloud.device)
                downsampled_points = torch.cat([downsampled_points, padding], dim=0)

        return downsampled_points.to('cpu')      
    
    def voxel_downsampling_open3d(self, points:np.array, voxel_size=0.1, num_points=50_000):
        # Convert NumPy array to Open3D PointCloud
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)

        # Apply voxelization
        voxelized_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        voxelized_points = np.asarray(voxelized_pcd.points)

        # Resample to ensure the desired number of points
        if num_points is not None:
            if len(voxelized_points) > num_points:
                indices = np.random.choice(len(voxelized_points), num_points, replace=False)
                voxelized_points = voxelized_points[indices]
            elif len(voxelized_points) < num_points:
                additional_indices = np.random.choice(len(voxelized_points), num_points - len(voxelized_points), replace=True)
                voxelized_points = np.vstack((voxelized_points, voxelized_points[additional_indices]))

        # Convert to PyTorch tensor
        torch_tensor = torch.tensor(voxelized_points, dtype=torch.float32)
        # final_num_points = torch_tensor.shape[0]
        return torch_tensor    
    
    def obtain_valid_lidar_points(self, projected_voxel_2d:torch.tensor, image_resize:tuple, 
                                  grid_sizes:Iterable[tuple]):
        
        valid_lidar_points_dict = {}
        
        W_img, H_img = image_resize        
        
        for grid_size in grid_sizes:
            W_grid, H_grid = grid_size
            # valid_lidar_points_dict[(W_grid, H_grid)] = {}
            
            grid_coords = projected_voxel_2d.clone()            
            grid_coords = torch.floor(grid_coords).long() # Discretize to grid cell indices
            
            # grid_coords = grid_coords[nonzero_depth_mask]
                        
            #Normalize to grid dimensions
            grid_coords[:, 0] = (grid_coords[:, 0] / W_img) * W_grid # X to grid width
            grid_coords[:, 1] = (grid_coords[:, 1] / H_img) * H_grid # Y to grid width                         
            
            valid_mask_x = (grid_coords[:, 0] >=0) & (grid_coords[:, 0] < W_grid)
            valid_mask_y = (grid_coords[:, 1] >=0) & (grid_coords[:, 1] < H_grid)             
            
            valid_grid_coords = grid_coords[valid_mask_x & valid_mask_y]                  

            # Map 2D grid coordinates to 1D indices for counting
            # Convert 2D grid indices (𝑥,𝑦) (x,y) to a 1D index using the formula: 
            # index = y * W_grid + x.
            linear_indices = valid_grid_coords[:, 1] * W_grid + valid_grid_coords[:, 0]

            # Count points per grid cell using scatter_add
            point_counts = torch.zeros(W_grid * H_grid, dtype=torch.int64)
            point_counts.scatter_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.int64))

            # Reshape to grid dimensions
            point_counts = point_counts.view(H_grid, W_grid)  
            # point_counts[point_counts  == 0] = 1e-5         
            
            # print(f'{grid_size} {point_counts[point_counts == 0].sum()}')   
            
            # valid_indices = valid_mask_x & valid_mask_y
            # print(f'{grid_size} {valid_indices[valid_indices == 0].sum()}')   
            
            valid_lidar_points_dict[grid_size] = {
                'valid_mask_x':valid_mask_x,
                'valid_mask_y':valid_mask_y,
                "valid_indices": valid_mask_x & valid_mask_y, #(N,) bool
                "valid_grid_coords":valid_grid_coords, #(2, N)
                "count_grid":point_counts
            }            

        return valid_lidar_points_dict

    def transform_camera_to_lidar_box3d(self, bboxes_3d:list, calibration_dict:dict):        
        #TODO, include P2, V2C as inv_rigid_transform, R0_rect
        bboxes_lidar = []

        r0_rect = np.zeros((4, 4))
        r0_rect[:3, :3] = calibration_dict['R0_rect'].reshape(3,3) #(3,3)
        r0_rect[3, 3] = 1

        r0_inv = np.linalg.inv(r0_rect) # shape (4, 4)        

        # inverse rigid transform
        v2c = calibration_dict['Tr_velo_to_cam'].reshape(3,4)
        c2v = np.zeros_like(v2c)
        c2v[0:3, 0:3] = np.transpose(v2c[0:3, 0:3])
        c2v[0:3, 3] = np.dot(-np.transpose(v2c[0:3, 0:3]), v2c[0:3, 3]) # shape (3, 4)                

        for box in bboxes_3d:
            x_cam, y_cam, z_cam, h, w, l, ry, dist_to_cam = box

            # step 1: Convert center from rectified camera to LiDAR coordinates
            xyz_cam = np.array([x_cam, y_cam, z_cam, 1]) #(4,)
            xyz_cam_rect = np.matmul(r0_inv, xyz_cam) #(4,)

            xyz_lidar = np.matmul(c2v, xyz_cam_rect)
            x_l, y_l, z_l = xyz_lidar

            # step 2: Convert ry (camera yaw) to rz (LiDAR yaw)
            rz = -ry - np.pi / 2
            #Normalize rz to [-pi, pi]
            # rz = (rz + np.pi) % (2 * np.pi) - np.pi

            # Reorder dimensions: [x, y, z, w, l, h, rz]
            bboxes_lidar.append([x_l, y_l, z_l, h, w, l, rz])
                        
        return np.array(bboxes_lidar)

class Kitti2DObjectDetectDataset(Dataset):
    
    def __init__(self, lidar_dir:str, 
                calibration_dir:str=None, 
                left_image_dir:str=None, 
                right_image_dir:str=None,
                labels_dir:str=None, 
                dataset_type:str="train"):
        
        self.left_image_dir = left_image_dir 
        self.right_image_dir = right_image_dir
                        
        if not bool(self.left_image_dir) and not bool(self.right_image_dir):
            raise Exception(f'Both Left and Right Images cannot be {self.left_image_dir}')
            
        self.calibration_dir = calibration_dir
        self.lidar_dir = lidar_dir
        self.labels_dir = labels_dir 
        self.dataset_type = dataset_type
                
        self.lidar_files = sorted(os.listdir(self.lidar_dir))
        self.calibration_files = sorted(os.listdir(self.calibration_dir)) if self.calibration_dir else None
        
        self.left_image_files = sorted(os.listdir(self.left_image_dir)) if self.left_image_dir else None
        self.right_image_files = sorted(os.listdir(self.right_image_dir)) if self.right_image_dir else None
        self.label_files = sorted(os.listdir(self.labels_dir)) if self.labels_dir else None
  
    def __len__(self):
        return len(
            self.lidar_files
        )
        
    def __getitem__(self, idx):
        
        lidar_file = self.lidar_files[idx]
        _id = lidar_file.split('.')[0]
        
        return {
            'lidar_file_path':f'{self.lidar_dir}/{lidar_file}',
            'calibration_file_path':f'{self.calibration_dir}/{_id}.txt',
            'left_image_file_path': f'{self.left_image_dir}/{_id}.png' if self.left_image_dir is not None else None,
            'right_image_file_path':f'{self.right_image_dir}/{_id}.png' if self.right_image_dir is not None else None,
            'label_file_path':f'{self.labels_dir}/{_id}.txt' if self.labels_dir is not None else None
        }
        
class KittiLidarFusionCollateFn(object):
    
    def __init__(self, image_resize:list, original_size:tuple=(1242, 375),
                precomputed_voxel_dir:str=None, precomputed_proj2d_dir:str=None,
                transformation=None, clip_distance:float=2.0, apply_augmentation:bool=True,
                project_2d:bool=False, voxelization:bool=False, 
                categorize_labels:bool=False):        
        
        self.image_resize = image_resize
        self.transformation = transformation
        self.clip_distance = clip_distance
        self.project_2d = project_2d
        self.voxelization = voxelization
        self.original_size = original_size
        
        self.original_width = self.original_size[0]
        self.original_height = self.original_size[1]
        
        self.resized_width = self.image_resize[0]
        self.resized_height = self.image_resize[1]
        
        self.voxelization_backend = "open3d"
        self.categorize_labels = categorize_labels
        
        self.precomputed_voxel_dir = precomputed_voxel_dir if precomputed_voxel_dir is not None else ""
        self.precomputed_proj2d_dir = precomputed_proj2d_dir if precomputed_proj2d_dir is not None else ""
        
        self.grid_sizes = [(13, 13), (26, 26), (52, 52)]        
        
        if self.transformation is None:
            self.transformation = albumentations.Compose(
                # [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)],
                [albumentations.LongestMaxSize(max_size=max(self.image_resize)),
                albumentations.PadIfNeeded(
                    min_height=self.image_resize[0],
                    min_width=self.image_resize[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )],                
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
            
            self.image_only_transformation = albumentations.Compose(
                [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)]
            )
            
        self.apply_augmentation = apply_augmentation
        self.image_augmentor = albumentations.Compose([
            albumentations.OneOf([
                albumentations.GaussianBlur(blur_limit=(1, 3), p=0.5),
                albumentations.MedianBlur(blur_limit=3, p=0.5),
                albumentations.MotionBlur(blur_limit=3, p=0.5)
            ], p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            ], p=0.5),
            albumentations.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
            albumentations.OneOf([
                albumentations.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, fill_value=0, p=0.5),
                albumentations.CropAndPad(percent=(-0.1, 0.1), pad_mode=cv2.BORDER_CONSTANT, p=0.5),
            ], p=0.5)
        ])                    
        
    def read_calibration_file(self, calib_file_path):
        calibration_dict = {}
        
        with open(calib_file_path, 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    key, value = line.split(':')
                    calibration_dict[key.strip()] = np.fromstring(
                        value, sep=' '
                    )
                    
        return calibration_dict

    def categorize_label_difficulty(self, truncation, occlusion):
        # Easy: Fully visible, truncation <= 15%
        if occlusion == 0 and truncation <= 0.15:
            return 0  # Easy

        # Moderate: Partly occluded, truncation <= 30%
        elif occlusion <= 1 and truncation <= 0.30:
            return 1  # Moderate

        # Hard: Difficult to see, truncation <= 50%
        elif occlusion <= 2 and truncation <= 0.50:
            return 2  # Hard
        
        return 2        
    
    def read_label_file(self, label_file_path:str):
        '''
        #Values    Index    Name      Description
        ----------------------------------------------------------------------------
        1        0       type      Describes the type of object: 'Car', 'Van', 'Truck',
                                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                    'Misc' or 'DontCare'
        1        1       truncated Float from 0 (non-truncated) to 1 (truncated), where
                                    truncated refers to the object leaving image boundaries
        1        2       occluded  Integer (0,1,2,3) indicating occlusion state:
                                    0 = fully visible, 1 = partly occluded
                                    2 = largely occluded, 3 = unknown
        1        3       alpha     Observation angle of object, ranging [-pi..pi]
        4        4-7       bbox      2D bounding box of object in the image (0-based index):
                                    contains left, top, right, bottom pixel coordinates
        3        8-10       dimensions 3D object dimensions: height, width, length (in meters)
        3        11-13       location  3D object location x,y,z in camera coordinates (in meters)
        1        14       rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1        15       score     Only for results: Float, indicating confidence in
                                    detection, needed for p/r curves, higher is better.

        '''
        class_labels = []
        bboxes = []
        categories = []
        
        with open(label_file_path, 'r') as file:
            for line in file:                
                parts = line.strip().split() # Split the line into parts

                # Extract class label and bounding box coordinates
                obj_type = parts[0]  # First element is the object type
                
                if Enums.mapping_dict and obj_type in Enums.mapping_dict:
                    obj_type = Enums.mapping_dict[obj_type]
                
                elif obj_type not in Enums.KiTTi_label2Id:
                    continue  # Skip invalid object types

                # Bounding box coordinates
                left = float(parts[4])  # left
                top = float(parts[5])  # top
                right = float(parts[6])  # right
                bottom = float(parts[7])  # bottom

                class_id = Enums.KiTTi_label2Id[obj_type]
                
                class_labels.append(class_id)
                bboxes.append([left, top, right, bottom])  

                if self.categorize_labels:
                    difficulty = self.categorize_label_difficulty(
                        float(parts[1]), int(parts[2])
                    )
                    
                    # categories[difficulty] = (class_id, [left, top, right, bottom])
                    categories.append(difficulty)               
        if self.categorize_labels:
            return class_labels, bboxes, categories
        else:
            return class_labels, bboxes                     
                
        # return class_labels, bboxes         

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
    
    def prepare_targets(self, batch_idx:int, class_labels:list, class_bboxes:list, letter_box:bool=True):

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

    def create_maps(self, projected_points:np.array, y_max:int, x_max:int, intensities:np.array, depths:np.array, heights:np.array):

        reflectance_map = np.zeros((y_max, x_max), dtype=np.float32)
        depth_map = np.zeros((y_max, x_max), dtype=np.float32)       
        
        for i in range(len(projected_points)):
            x, y = int(projected_points[i, 0]), int(projected_points[i, 1])
            if 0 <= x < x_max and 0 <= y < y_max:
                # Use maximum intensity for reflectance
                reflectance_map[y, x] = max(reflectance_map[y, x], intensities[i])
                # Use the closest depth value
                if depth_map[y, x] == 0:  # if depth is not set yet
                    depth_map[y, x] = depths[i]        

        reflectance_map = (reflectance_map / np.max(reflectance_map) * 255).astype(np.uint8)
        depth_map = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
        
        return depth_map, reflectance_map

    def preprocess(self, lidar_point_cloud:np.array, calibration_dict:dict, image_array:np.array):
        
        points_2d, depths,_ = LidarPreprocessorUtils().transform_lidar_points(
            lidar_point_cloud, calibration_dict
        )        
        
        x_min, y_min = 0, 0
        x_max, y_max = image_array.shape[1], image_array.shape[0]        
        
        fov_inds = (
                (points_2d[:, 0] < x_max)
                & (points_2d[:, 0] >= x_min)
                & (points_2d[:, 1] < y_max)
                & (points_2d[:, 1] >= y_min)
        )    
        
        fov_inds = fov_inds & (
                    lidar_point_cloud[:, 0] > self.clip_distance)      
        
        projected_points = points_2d[fov_inds]
        heights = lidar_point_cloud[fov_inds, 2]
        
        depth_map, reflectance_map = self.create_maps(
            projected_points, y_max, x_max, 
            lidar_point_cloud[fov_inds, -1],
            lidar_point_cloud[fov_inds, -2],
            heights            
        )         
        
        return depth_map, reflectance_map, projected_points
    
    def preprocess_yolo_inputs(self, lidar_point_cloud:np.array, calibration_dict:dict, 
                               left_image_arr=None, right_image_arr=None):
        if left_image_arr is not None:
            depth_map, reflectance_map, _ = self.preprocess(lidar_point_cloud, calibration_dict, left_image_arr)
            depth_map, reflectance_map = np.expand_dims(depth_map, axis=-1), np.expand_dims(reflectance_map, axis=-1)            
            
            combined_image = np.concatenate([
                left_image_arr, depth_map, reflectance_map
            ], axis=-1)

        elif right_image_arr is not None:
            depth_map, reflectance_map, _ = self.preprocess(lidar_point_cloud, calibration_dict, right_image_arr)  
            depth_map, reflectance_map = np.expand_dims(depth_map, axis=-1), np.expand_dims(reflectance_map, axis=-1)
            
            combined_image = np.concatenate([
                right_image_arr, depth_map, reflectance_map
            ], axis=-1)                  
            
        if self.apply_augmentation:
            combined_image = self.image_augmentor(image=combined_image)['image']                
                
        return combined_image

    def preprocess_pointnet_inputs(self, lidar_point_cloud:np.array, lidar_file_path:str,
                                   calibration_dict:dict):

        lidar_fn = lidar_file_path.split('/')[-1]
        _id = lidar_fn.split('.')[0]           
        voxel_file_path = f'{self.precomputed_voxel_dir}/{_id}_voxelized.bin'

        if os.path.exists(voxel_file_path):                        
            voxelized_point_cloud = np.fromfile(voxel_file_path,
                                                dtype=np.float32).reshape(-1, 3)
            voxelized_point_cloud = torch.from_numpy(voxelized_point_cloud)
            voxelized_point_cloud = voxelized_point_cloud.transpose(1, 0)

        else:
            voxelized_point_cloud = torch.tensor(lidar_point_cloud)   
            if self.voxelization_backend == "torch":
                voxelized_point_cloud = LidarPreprocessorUtils().voxel_downsampling_torch(torch.from_numpy(lidar_point_cloud), 
                                                                    voxel_size=0.2,
                                                                    num_points=50_000)
            if self.voxelization_backend == "open3d":
                voxelized_point_cloud = LidarPreprocessorUtils().voxel_downsampling_open3d(lidar_point_cloud[:, :3], 
                                                                    voxel_size=0.2,
                                                                    num_points=50_000)
            
            voxelized_point_cloud = voxelized_point_cloud[:, :3].transpose(1, 0)            

        valid_lidar_points_dict = {}
        
        if self.project_2d:
            if os.path.exists(f'{self.precomputed_proj2d_dir}/{_id}'):
                fn = f'{self.precomputed_proj2d_dir}/{_id}'
                
                for grid_size in self.grid_sizes:
                    valid_indices = np.fromfile(
                        f'{fn}/{grid_size}_valid_indices.bin',
                        dtype=np.int8
                    ).astype(bool)                            
                    
                    valid_grid_coords = np.fromfile(
                        f'{fn}/{grid_size}_valid_grid_coords.bin',
                        dtype=np.int64
                    ).astype(np.int64).reshape(-1, 2)                                                      

                    grid_coords = np.fromfile(
                        f'{fn}/{grid_size}_grid_coords.bin',
                        dtype=np.int64
                    ).astype(np.int64).reshape(-1, 2)                                                      

                    valid_lidar_points_dict[grid_size] = {
                        'valid_indices':torch.from_numpy(valid_indices),
                        "valid_grid_coords":torch.from_numpy(valid_grid_coords),
                        "grid_coords":torch.from_numpy(grid_coords)
                    }
                    
            else:
                projected_voxel_2d, depth, negative_depth_mask = LidarPreprocessorUtils().transform_lidar_points(
                    voxelized_point_cloud.transpose(0, 1).cpu().numpy(), calibration_dict
                )
                
                
                projected_voxel_2d[:, 0] = (projected_voxel_2d[:, 0]/ self.original_size[0]) * (self.image_resize[0])
                projected_voxel_2d[:, 1] = (projected_voxel_2d[:, 1]/ self.original_size[1]) * (self.image_resize[1])                            
                                
                depth[negative_depth_mask] = 1e-5
                projected_voxel_2d[:, 0] /= depth
                projected_voxel_2d[:, 1] /= depth
                
                projected_voxel_2d = torch.from_numpy(projected_voxel_2d)
                valid_lidar_points_dict = LidarPreprocessorUtils().obtain_valid_lidar_points(
                    projected_voxel_2d, self.image_resize, self.grid_sizes
                )
                
        return valid_lidar_points_dict, voxelized_point_cloud       

    def __call__(self, batch_data_filepaths:List[Dict]):

        batch_data_items = {
            "images": [], #list of tensors --> stack --> tensor (bs, n_c, h, w),
            "targets": [],
            "image_paths":[],
            "raw_point_clouds":[],
            "proj2d_pc_mask":[], 
            'target_difficulty':[]
            # "bboxes":[], # list of list of list [bs * [num_labels * [x,y,x,y] ]] (inner list of 4 elements)
            # "class_labels":[] # list of list of class_labels [bs * [num_labels] ] (inner list is class_ids)
        }        
        
        for idx, file_path_dict in enumerate(batch_data_filepaths):
            lidar_file_path = file_path_dict['lidar_file_path']
            calibration_file_path = file_path_dict['calibration_file_path']
            left_image_file_path = file_path_dict['left_image_file_path']
            right_image_file_path = file_path_dict['right_image_file_path']
            label_file_path = file_path_dict['label_file_path']            
            
            if os.path.exists(calibration_file_path):
                calibration_dict = self.read_calibration_file(calibration_file_path)
            else:
                print(f'Calib {calibration_file_path} Does not Exist!')
                
            left_image_arr = None
            right_image_arr = None                

            if bool(left_image_file_path) and os.path.exists(left_image_file_path):
                left_image_arr = cv2.imread(left_image_file_path)
                batch_data_items['image_paths'].append(left_image_file_path)
            
            if bool(right_image_file_path) and os.path.exists(right_image_file_path):
                right_image_arr = cv2.imread(right_image_file_path)    
                batch_data_items['image_paths'].append(right_image_file_path)
                
            if left_image_arr is None and right_image_arr is None:
                print(f'Left Image Path {left_image_file_path} and Right Image Path {right_image_file_path}')
                exit(1)
                
            if os.path.exists(lidar_file_path):
                lidar_point_cloud = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)
                
                combined_image = self.preprocess_yolo_inputs(
                    lidar_point_cloud, calibration_dict, 
                    left_image_arr, right_image_arr
                )    
                
                valid_lidar_points_dict, voxelized_point_cloud = self.preprocess_pointnet_inputs(
                    lidar_point_cloud, lidar_file_path, calibration_dict
                )                
                
                if label_file_path is not None:
                    if os.path.exists(label_file_path):
                        # class_labels, label_bboxes = self.read_label_file(label_file_path)

                        if self.categorize_labels:
                            class_labels, label_bboxes, target_difficulty = self.read_label_file(label_file_path) 

                            batch_data_items['target_difficulty'].append(
                                torch.tensor(target_difficulty, dtype=torch.uint8)
                            )
                            
                        else:
                            class_labels, label_bboxes = self.read_label_file(label_file_path)
                         
                        transformed_dict = self.transform_sample(
                            combined_image, label_bboxes, class_labels
                        )      
                        
                        targets = self.prepare_targets(
                            idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                        )                                                      
                        
                        batch_data_items['targets'].append(
                            torch.tensor(targets, dtype=torch.float32)
                        )
                        
                    else:
                        print(f'Label File Path not found!!')
                        exit(1)                    
                        
                else:
                    transformed_dict = self.transform_sample(
                        combined_image
                    )                                                

                image_tensor = torch.from_numpy(transformed_dict['image']).permute((2, 0, 1))                          
                batch_data_items['images'].append(image_tensor)
                batch_data_items['proj2d_pc_mask'].append(valid_lidar_points_dict) 
                batch_data_items['raw_point_clouds'].append(voxelized_point_cloud)                

                
            else:
                ''' 
                TODO, add RGB only inputs preprocessing here. 
                '''
                print(f'Lidar {lidar_file_path} Does not Exist!')
                exit(1)            
                
        batch_data_items['images'] = torch.stack(
            batch_data_items['images'], dim=0
        ).float()
        
        batch_data_items['raw_point_clouds'] = torch.stack(
            batch_data_items['raw_point_clouds'], dim=0
        )
        
        if batch_data_items['targets']:
            batch_data_items['targets'] = torch.concat(
                batch_data_items['targets'], dim=0
            )
            
        if batch_data_items['target_difficulty']:
            batch_data_items['target_difficulty'] = torch.concat(
                batch_data_items['target_difficulty'], dim=0
            )            
            
        return batch_data_items
        
class KittiMLSFCollateFn(object):
    
    def __init__(self, image_resize:list, detection_head:str,
                original_size:tuple=(375, 1242), lidar_map_type:str='depth_map',
                transformation=None, apply_augmentation=False):
        
        self.image_resize = image_resize
        self.original_size = original_size
        self.detection_head = detection_head
        
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
                [
                    albumentations.LongestMaxSize(max_size=max(self.image_resize)),
                    albumentations.PadIfNeeded(
                        min_height=self.resized_height,
                        min_width=self.resized_width,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0
                    ),
                    albumentations.CenterCrop(
                        height=self.resized_height,
                        width=self.resized_width,
                        always_apply=True
                    )
                ]
            )

        # Front side (of vehicle) Point Cloud boundary for BEV
        self.boundary_front = {
            "minX": 0,
            "maxX": 50,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }

        self.bev_grid_x_res = (self.boundary_front['maxX'] - self.boundary_front['minX'])/self.resized_width
        self.bev_grid_y_res = (self.boundary_front['maxY'] - self.boundary_front['minY'])/self.resized_height

        # Back back (of vehicle) Point Cloud boundary for BEV
        self.boundary_back = {
            "minX": -50,
            "maxX": 0,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }        

    def read_calibration_file(self, calib_file_path):

        calibration_dict = {}
        with open(calib_file_path, 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    key, value = line.split(':')
                    calibration_dict[key.strip()] = np.fromstring(
                        value, sep=' '
                    )
                    
        return calibration_dict

    def read_label_file(self, label_file_path:str):
        '''
        #Values    Index    Name      Description
        ----------------------------------------------------------------------------
        1        0       type      Describes the type of object: 'Car', 'Van', 'Truck',
                                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                    'Misc' or 'DontCare'
        1        1       truncated Float from 0 (non-truncated) to 1 (truncated), where
                                    truncated refers to the object leaving image boundaries
        1        2       occluded  Integer (0,1,2,3) indicating occlusion state:
                                    0 = fully visible, 1 = partly occluded
                                    2 = largely occluded, 3 = unknown
        1        3       alpha     Observation angle of object, ranging [-pi..pi]
        4        4-7       bbox      2D bounding box of object in the image (0-based index):
                                    contains left, top, right, bottom pixel coordinates
        3        8-10       dimensions 3D object dimensions: height, width, length (in meters)
        3        11-13       location  3D object location x,y,z in camera coordinates (in meters)
        1        14       rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1        15       score     Only for results: Float, indicating confidence in
                                    detection, needed for p/r curves, higher is better.

        '''
        class_labels = []
        bboxes_2d = []
        bboxes_3d = []
        
        with open(label_file_path, 'r') as file:
            for line in file:                
                parts = line.strip().split() # Split the line into parts

                # Extract class label and bounding box coordinates
                obj_type = parts[0]  # First element is the object type

                if Enums.mapping_dict and obj_type in Enums.mapping_dict:
                    obj_type = Enums.mapping_dict[obj_type]

                elif obj_type not in Enums.KiTTi_label2Id:
                    continue  # Skip invalid object types

                # 2D Bounding box coordinates
                # ⚠️ KITTI defines height along the Y-axis, width along X, and length along Z in camera coordinates.
                left = float(parts[4])  # left
                top = float(parts[5])  # top
                right = float(parts[6])  # right
                bottom = float(parts[7])  # bottom            

                # 3D Bounding Box Coordinates
                h = float(parts[8]) #(in meters, vertical dimension)
                w = float(parts[9]) #(lateral dimension)
                l = float(parts[10]) #(along vehicle/driving direction)
                
                # 3D center of the object in camera coordinate space
                # x – left/right; y – vertical; z – depth (forward from camera
                t = float(parts[11]), float(parts[12]), float(parts[13])
                
                # Euclidean distance from the camera center to the 3D object center.
                dist_to_cam = np.linalg.norm(t)
                
                # rotation angle ry of the object in camera coordinates
                ry = float(parts[14])

                if self.detection_head == 'yolo':
                    class_id = Enums.KiTTi_label2Id[obj_type]
                elif self.detection_head == 'ssd': #background ID present for SSD
                    class_id = Enums.KiTTi_label2Id_SSD[obj_type]
                
                class_labels.append(class_id)
                bboxes_2d.append([left, top, right, bottom])
                
                #FIXME Complete, rearranged. (x, y, z, h, w, l, ry) 
                bboxes_3d.append([t[0], t[1], t[2], h, w, l, ry, dist_to_cam])
                           
        return class_labels, bboxes_2d, bboxes_3d 

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
    
    def prepare_targets_3d(self, batch_idx: int, class_labels: list, class_bboxes: list):
        #TODO, fix based on the order of bboxes elements after cam2lidar.
        targets = []
        for bbox, class_id in zip(class_bboxes, class_labels):
            x_l, y_l, z_l, h, w, l, yaw = bbox[:]

            # Slight adjustment for small objects
            l = l + 0.3
            w = w + 0.3

            # Reverse yaw to align with LiDAR coordinate frame
            yaw = 2 * np.pi - yaw

            if (x_l > self.boundary_front['minX']) and (x_l < self.boundary_front['maxX']) and \
            (y_l > self.boundary_front['minY']) and (y_l < self.boundary_front['maxY']):
                
                x_l = (x_l - self.boundary_front['minX']) / (self.boundary_front['maxX'] - self.boundary_front['minX'])
                y_l = (y_l - self.boundary_front['minY']) / (self.boundary_front['maxY'] - self.boundary_front['minY'])

                w = w / (self.boundary_front['maxY'] - self.boundary_front['minY'])
                l = l / (self.boundary_front['maxX'] - self.boundary_front['minX'])

                sin_yaw = np.sin(yaw)
                cos_yaw = np.cos(yaw)

                targets.append([batch_idx, class_id, y_l, x_l, w, l, sin_yaw, cos_yaw])
        
        return targets

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
    
    def generate_bev_map(self, lidar_point_cloud: np.array):
        
        # Remove the point out of range x,y,z
        mask = np.where((lidar_point_cloud[:, 0] >= self.boundary_front['minX']) & (lidar_point_cloud[:, 0] <= self.boundary_front['maxX']) & (lidar_point_cloud[:, 1] >= self.boundary_front['minY']) & (
                lidar_point_cloud[:, 1] <= self.boundary_front['maxY']) & (lidar_point_cloud[:, 2] >= self.boundary_front['minZ']) & (lidar_point_cloud[:, 2] <= self.boundary_front['maxZ']))
        lidar_point_cloud = lidar_point_cloud[mask]

        lidar_point_cloud[:, 2] = lidar_point_cloud[:, 2] - self.boundary_front['minZ']

        DISCRETIZATION = (self.boundary_front["maxX"] - self.boundary_front["minX"])/self.resized_height
        
        Height = self.resized_height + 1
        Width = self.resized_width + 1

        # Discretize Feature Map
        PointCloud = np.copy(lidar_point_cloud)
        PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION))
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2)

        # sort-3times
        indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
        PointCloud = PointCloud[indices]

        # Height Map
        heightMap = np.zeros((Height, Width))

        _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
        PointCloud_frac = PointCloud[indices]
        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(self.boundary_front['maxZ'] - self.boundary_front['minZ']))
        heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

        # Intensity Map & DensityMap
        intensityMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        PointCloud_top = PointCloud[indices]

        normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

        intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, Height - 1, Width - 1))
        RGB_Map[2, :, :] = densityMap[:self.resized_height, :self.resized_width]  # r_map
        RGB_Map[1, :, :] = heightMap[:self.resized_height, :self.resized_width]  # g_map
        RGB_Map[0, :, :] = intensityMap[:self.resized_height, :self.resized_width]  # b_map

        return RGB_Map

    def __call__(self, batch_data_filepaths:List[Dict]):

        batch_data_items = {
            'images':[],
            "image_paths":[],
            'lidar_2d':[],
            "targets": [],
            "targets_3d":[], 
            "label_file_path":[]
        }

        for idx, file_path_dict in enumerate(batch_data_filepaths):
            lidar_file_path = file_path_dict['lidar_file_path']
            calibration_file_path = file_path_dict['calibration_file_path']
            left_image_file_path = file_path_dict['left_image_file_path']
            right_image_file_path = file_path_dict['right_image_file_path']
            label_file_path = file_path_dict['label_file_path']

            if bool(left_image_file_path) and os.path.exists(left_image_file_path):
                left_image_arr = cv2.imread(left_image_file_path)
                batch_data_items['image_paths'].append(left_image_file_path)

            if bool(right_image_file_path) and os.path.exists(right_image_file_path):
                right_image_arr = cv2.imread(right_image_file_path)    
                batch_data_items['image_paths'].append(right_image_file_path)

            if left_image_arr is None and right_image_arr is None:
                print(f'Left Image Path {left_image_file_path} and Right Image Path {right_image_file_path}')
                exit(1)

            if os.path.exists(lidar_file_path):
                lidar_point_cloud = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)

                if self.lidar_map_type == 'depth_map':
                    lidar_map = self.lidar_to_depth_map(lidar_point_cloud)
                elif self.lidar_map_type == 'bev_map':
                    lidar_map = self.generate_bev_map(lidar_point_cloud)

            else:
                print(f'Lidar {lidar_file_path} Does not Exist!')
                exit(1)

            if os.path.exists(calibration_file_path):
                calibration_dict = self.read_calibration_file(calibration_file_path)
            else:
                print(f'Calib {calibration_file_path} Does not Exist!')
                calibration_dict = {}

            if label_file_path is not None:
                if os.path.exists(label_file_path):
                    #label_bboxes_3d [t[0], t[1], t[2], h, w, l, ry, dist_to_cam]
                    class_labels, label_bboxes_2d, label_bboxes_3d = self.read_label_file(label_file_path)
                    batch_data_items['label_file_path'].append(label_file_path)

                    #left_image_arr : (375, 1242, 3)                 
                    transformed_dict = self.transform_sample(
                        left_image_arr, label_bboxes_2d, class_labels
                    )

                    targets = self.prepare_targets_2d(
                        idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                    ) #prepared targets within (0 to 1) normalized, with image_resize

                    batch_data_items['targets'].append(
                        torch.tensor(targets, dtype=torch.float32)
                    )

                    if calibration_dict and label_bboxes_3d:
                        #targets_3d [x_l, y_l, z_l, h, w, l, rz]
                        targets_3d = LidarPreprocessorUtils().transform_camera_to_lidar_box3d(label_bboxes_3d, calibration_dict)

                        # [batch_idx, class_id, y_l, x_l, w, l, sin_yaw, cos_yaw]
                        targets_3d = self.prepare_targets_3d(idx, class_labels, targets_3d)

                        batch_data_items['targets_3d'].append(
                            torch.tensor(targets_3d, dtype=torch.float32)
                        )
                        
                        # bev_image = draw_bev_with_boxes(lidar_map, targets_3d, self.boundary_front, 
                        #                 self.bev_grid_x_res, self.bev_grid_y_res)

                        # print(label_file_path)
                        # print(targets_3d)
                        # cv2.imwrite("bev_with_boxes.png", bev_image)
                        # exit(1)                        

                    left_image = transformed_dict['image']
                    if self.lidar_map_type == 'depth_map':
                        lidar_map = self.transform_sample(lidar_map)['image']

                else:
                    print(f'Label File Path not found!!')
                    exit(1)

            else:
                left_image = self.transform_sample(
                    left_image_arr
                )['image']
                
                if self.lidar_map_type == 'depth_map':
                    lidar_map = self.transform_sample(lidar_map)['image']

            image_tensor = torch.from_numpy(left_image).permute((2, 0, 1)) #(nc, h, w)

            if self.lidar_map_type == 'depth_map':
                lidar_map_tensor = torch.from_numpy(lidar_map).permute((2, 0, 1)) #(nc, h, w)
            elif self.lidar_map_type == 'bev_map':
                lidar_map_tensor = torch.from_numpy(lidar_map) #(nc, h, w)

            batch_data_items['images'].append(image_tensor)
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

            batch_data_items['targets_3d'] = torch.concat(
                batch_data_items['targets_3d'], dim=0
            )            

        return batch_data_items

class KittiMLSFMobilenet(object):

    def __init__(self, image_resize:list, detection_head:str,
                original_size:tuple=(375, 1242), lidar_map_type:str='depth_map',
                transformation=None, apply_augmentation=False):

        self.image_resize = image_resize
        self.original_size = original_size
        self.detection_head = detection_head
        
        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]
        
        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]
    
        self.lidar_map_type = lidar_map_type
        self.apply_augmentation = apply_augmentation
        
        # Front side (of vehicle) Point Cloud boundary for BEV
        self.boundary_front = {
            "minX": 0,
            "maxX": 50,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }

        self.bev_grid_x_res = (self.boundary_front['maxX'] - self.boundary_front['minX'])/self.resized_width
        self.bev_grid_y_res = (self.boundary_front['maxY'] - self.boundary_front['minY'])/self.resized_height

        # Back back (of vehicle) Point Cloud boundary for BEV
        self.boundary_back = {
            "minX": -50,
            "maxX": 0,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }           
        
        self.transformation = albumentations.Compose(
                # [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)],
                [albumentations.LongestMaxSize(max_size=max(self.image_resize)),
                albumentations.PadIfNeeded(
                    min_height=self.resized_height,
                    min_width=self.resized_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ), 
                albumentations.CenterCrop(height=self.resized_height, width=self.resized_width, always_apply=True)],
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        
        self.image_only_transformation = albumentations.Compose(
            [albumentations.Resize(height=self.resized_height, width=self.resized_width, always_apply=True)]
        )
        
    def read_calibration_file(self, calib_file_path):

        calibration_dict = {}
        with open(calib_file_path, 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    key, value = line.split(':')
                    calibration_dict[key.strip()] = np.fromstring(
                        value, sep=' '
                    )
                    
        return calibration_dict

    def read_label_file(self, label_file_path:str):
        '''
        #Values    Index    Name      Description
        ----------------------------------------------------------------------------
        1        0       type      Describes the type of object: 'Car', 'Van', 'Truck',
                                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                    'Misc' or 'DontCare'
        1        1       truncated Float from 0 (non-truncated) to 1 (truncated), where
                                    truncated refers to the object leaving image boundaries
        1        2       occluded  Integer (0,1,2,3) indicating occlusion state:
                                    0 = fully visible, 1 = partly occluded
                                    2 = largely occluded, 3 = unknown
        1        3       alpha     Observation angle of object, ranging [-pi..pi]
        4        4-7       bbox      2D bounding box of object in the image (0-based index):
                                    contains left, top, right, bottom pixel coordinates
        3        8-10       dimensions 3D object dimensions: height, width, length (in meters)
        3        11-13       location  3D object location x,y,z in camera coordinates (in meters)
        1        14       rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1        15       score     Only for results: Float, indicating confidence in
                                    detection, needed for p/r curves, higher is better.

        '''
        class_labels = []
        bboxes_2d = []
        bboxes_3d = []
        
        with open(label_file_path, 'r') as file:
            for line in file:                
                parts = line.strip().split() # Split the line into parts

                # Extract class label and bounding box coordinates
                obj_type = parts[0]  # First element is the object type
                
                if Enums.mapping_dict and obj_type in Enums.mapping_dict:
                    obj_type = Enums.mapping_dict[obj_type]
                
                elif obj_type not in Enums.KiTTi_label2Id:
                    continue  # Skip invalid object types

                # 2D Bounding box coordinates
                # ⚠️ KITTI defines height along the Y-axis, width along X, and length along Z in camera coordinates.
                left = float(parts[4])  # left
                top = float(parts[5])  # top
                right = float(parts[6])  # right
                bottom = float(parts[7])  # bottom            
                
                # 3D Bounding Box Coordinates
                h = float(parts[8]) #(in meters, vertical dimension)
                w = float(parts[9]) #(lateral dimension)
                l = float(parts[10]) #(along vehicle/driving direction)
                
                # 3D center of the object in camera coordinate space
                # x – left/right; y – vertical; z – depth (forward from camera
                t = float(parts[11]), float(parts[12]), float(parts[13])
                
                # Euclidean distance from the camera center to the 3D object center.
                dist_to_cam = np.linalg.norm(t)
                
                # rotation angle ry of the object in camera coordinates
                ry = float(parts[14])

                if self.detection_head == 'yolo':
                    class_id = Enums.KiTTi_label2Id[obj_type]
                elif self.detection_head == 'ssd': #background ID present for SSD
                    class_id = Enums.KiTTi_label2Id_SSD[obj_type]

                class_labels.append(class_id)
                bboxes_2d.append([left, top, right, bottom])

                #FIXME Complete, rearranged. (x, y, z, h, w, l, ry) 
                bboxes_3d.append([t[0], t[1], t[2], h, w, l, ry, dist_to_cam])

        return class_labels, bboxes_2d, bboxes_3d 

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
    
    def generate_bev_map(self, lidar_point_cloud: np.array):
        
        # Remove the point out of range x,y,z
        mask = np.where((lidar_point_cloud[:, 0] >= self.boundary_front['minX']) & (lidar_point_cloud[:, 0] <= self.boundary_front['maxX']) & (lidar_point_cloud[:, 1] >= self.boundary_front['minY']) & (
                lidar_point_cloud[:, 1] <= self.boundary_front['maxY']) & (lidar_point_cloud[:, 2] >= self.boundary_front['minZ']) & (lidar_point_cloud[:, 2] <= self.boundary_front['maxZ']))
        lidar_point_cloud = lidar_point_cloud[mask]

        lidar_point_cloud[:, 2] = lidar_point_cloud[:, 2] - self.boundary_front['minZ']

        DISCRETIZATION = (self.boundary_front["maxX"] - self.boundary_front["minX"])/self.resized_height
        
        Height = self.resized_height + 1
        Width = self.resized_width + 1

        # Discretize Feature Map
        PointCloud = np.copy(lidar_point_cloud)
        PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION))
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2)

        # sort-3times
        indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
        PointCloud = PointCloud[indices]

        # Height Map
        heightMap = np.zeros((Height, Width))

        _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
        PointCloud_frac = PointCloud[indices]
        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(self.boundary_front['maxZ'] - self.boundary_front['minZ']))
        heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

        # Intensity Map & DensityMap
        intensityMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        PointCloud_top = PointCloud[indices]

        normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

        intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, Height - 1, Width - 1))
        RGB_Map[2, :, :] = densityMap[:self.resized_height, :self.resized_width]  # r_map
        RGB_Map[1, :, :] = heightMap[:self.resized_height, :self.resized_width]  # g_map
        RGB_Map[0, :, :] = intensityMap[:self.resized_height, :self.resized_width]  # b_map

        return RGB_Map

    def __call__(self, batch_data_filepaths:List[Dict]):

        batch_data_items = {
            'images':[],
            "image_paths":[],
            'lidar_2d':[],
            "targets": []
        }

        for idx, file_path_dict in enumerate(batch_data_filepaths):
            lidar_file_path = file_path_dict['lidar_file_path']
            calibration_file_path = file_path_dict['calibration_file_path']
            left_image_file_path = file_path_dict['left_image_file_path']
            right_image_file_path = file_path_dict['right_image_file_path']
            label_file_path = file_path_dict['label_file_path']

            if bool(left_image_file_path) and os.path.exists(left_image_file_path):
                left_image_arr = cv2.imread(left_image_file_path)
                batch_data_items['image_paths'].append(left_image_file_path)

            if bool(right_image_file_path) and os.path.exists(right_image_file_path):
                right_image_arr = cv2.imread(right_image_file_path)    
                batch_data_items['image_paths'].append(right_image_file_path)

            if left_image_arr is None and right_image_arr is None:
                print(f'Left Image Path {left_image_file_path} and Right Image Path {right_image_file_path}')
                exit(1)

            if os.path.exists(lidar_file_path):
                lidar_point_cloud = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)

                if self.lidar_map_type == 'depth_map':
                    lidar_map = self.lidar_to_depth_map(lidar_point_cloud)
                elif self.lidar_map_type == 'bev_map':
                    lidar_map = self.generate_bev_map(lidar_point_cloud)

            else:
                print(f'Lidar {lidar_file_path} Does not Exist!')
                exit(1)

            if os.path.exists(calibration_file_path):
                calibration_dict = self.read_calibration_file(calibration_file_path)
            else:
                print(f'Calib {calibration_file_path} Does not Exist!')                

            if label_file_path is not None:
                if os.path.exists(label_file_path):
                    #label_bboxes_3d [t[0], t[1], t[2], h, w, l, ry, dist_to_cam]
                    class_labels, label_bboxes_2d, label_bboxes_3d = self.read_label_file(label_file_path)                    

                    #left_image_arr : (375, 1242, 3)                 
                    transformed_dict = self.transform_sample(
                        left_image_arr, label_bboxes_2d, class_labels
                    )

                    targets = self.prepare_targets_2d(
                        idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                    ) #prepared targets within (0 to 1) normalized, with image_resize

                    batch_data_items['targets'].append(
                        torch.tensor(targets, dtype=torch.float32)
                    )

                    left_image = transformed_dict['image']
                    if self.lidar_map_type == 'depth_map':
                        lidar_map = self.transform_sample(lidar_map)['image']
                        
            else:
                left_image = self.transform_sample(
                    left_image_arr
                )['image']

                lidar_map = self.transform_sample(lidar_map)['image']

            image_tensor = torch.from_numpy(left_image).permute((2, 0, 1)) #(nc, h, w)
            image_tensor = self.normalize_tensor(image_tensor)

            if self.lidar_map_type == 'depth_map':
                lidar_map_tensor = torch.from_numpy(lidar_map).permute((2, 0, 1)) #(nc, h, w)
            elif self.lidar_map_type == 'bev_map':
                lidar_map_tensor = torch.from_numpy(lidar_map) #(nc, h, w)

            lidar_map_tensor = self.normalize_tensor(lidar_map_tensor)
            
            batch_data_items['images'].append(image_tensor)
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
            
    def normalize_tensor(self, tensor:torch.Tensor):
        
        tensor = tensor.float() / 255.0 
        
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = (tensor - mean) / std
        return tensor

class KitiiMLSFCollateAugment(KittiMLSFCollateFn):

    def __init__(self, image_resize, detection_head, original_size = (375, 1242), transformation=None, apply_augmentation=False):
        super().__init__(image_resize, detection_head, original_size, transformation, apply_augmentation)
        
    def augment_lidar_point_cloud(self, point_cloud, 
                                translation_range_xy=(-5,5), translation_range_z=(-1,1),
                                scaling_range=(0.9,1.1), rotation_range=(-5,5)):
        """
        Augment the LiDAR point cloud.
        
        Args:
            point_cloud (np.array): shape (N,4) [x, y, z, intensity].
            translation_range_xy (tuple): translation range for x and y in meters.
            translation_range_z (tuple): translation range for z.
            scaling_range (tuple): uniform random scaling factor for each axis.
            rotation_range (tuple): rotation range (in degrees) about the z-axis.
        
        Returns:
            augmented_point_cloud, scale_factors, translation, rotation_rad
        """
        # Random scaling factors for x, y, z.
        scale_x = random.uniform(*scaling_range)
        scale_y = random.uniform(*scaling_range)
        scale_z = random.uniform(*scaling_range)
        scale_factors = np.array([scale_x, scale_y, scale_z])
        
        # Apply scaling.
        augmented = point_cloud.copy()
        augmented[:, :3] *= np.array([scale_x, scale_y, scale_z])
        
        # Random translation.
        trans_x = random.uniform(*translation_range_xy)
        trans_y = random.uniform(*translation_range_xy)
        trans_z = random.uniform(*translation_range_z)
        translation = np.array([trans_x, trans_y, trans_z])
        augmented[:, :3] += translation
        
        # Random rotation about the z-axis.
        angle_deg = random.uniform(*rotation_range)
        angle_rad = np.deg2rad(angle_deg)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        R = np.array([[cos_val, -sin_val, 0],
                    [sin_val,  cos_val, 0],
                    [0,       0,      1]])
        augmented[:, :3] = augmented[:, :3].dot(R.T)

        return augmented, scale_factors, translation, angle_rad

    def augment_camera_image(self, image, scaling_range=(0.9,1.1), translation_range=(-50,50)):
        """
        Augment the camera image.
        
        Args:
            image (np.array): The image in HxWxC.
            scaling_range (tuple): Range for random scaling factor.
            translation_range (tuple): Range for random translation in pixels.
        
        Returns:
            augmented_image, scale_factor, translation_pixels
        """
        h, w, _ = image.shape
        # Random scaling.
        scale = random.uniform(*scaling_range)
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled_img = cv2.resize(image, (new_w, new_h))
        
        # Random translation.
        tx = int(random.uniform(*translation_range))
        ty = int(random.uniform(*translation_range))
        M = np.float32([[1, 0, tx],
                        [0, 1, ty]])
        # Use warpAffine to shift image.
        translated_img = cv2.warpAffine(scaled_img, M, (new_w, new_h),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return translated_img, scale, (tx, ty)
    
    def __call__(self, batch_data_filepaths:List[Dict]):
        
        batch_data_items = {
            'images':[],
            "image_paths":[],
            'lidar_2d':[],
            'augmented_lidar_2d':[],
            "targets": []
        }
        
        for idx, file_path_dict in enumerate(batch_data_filepaths):
            lidar_file_path = file_path_dict['lidar_file_path']
            calibration_file_path = file_path_dict['calibration_file_path']
            left_image_file_path = file_path_dict['left_image_file_path']
            right_image_file_path = file_path_dict['right_image_file_path']
            label_file_path = file_path_dict['label_file_path']

            if bool(left_image_file_path) and os.path.exists(left_image_file_path):
                left_image_arr = cv2.imread(left_image_file_path)
                left_image_arr_augmented, _, _ = self.augment_camera_image(image=left_image_arr)
                batch_data_items['image_paths'].append(left_image_file_path)

            if bool(right_image_file_path) and os.path.exists(right_image_file_path):
                right_image_arr = cv2.imread(right_image_file_path)
                right_image_arr_augmented, _, _ = self.augment_camera_image(image=right_image_arr)
                batch_data_items['image_paths'].append(right_image_file_path)

            if left_image_arr is None and right_image_arr is None:
                print(f'Left Image Path {left_image_file_path} and Right Image Path {right_image_file_path}')
                exit(1)

            if os.path.exists(lidar_file_path):
                lidar_point_cloud = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)
                # lidar_point_cloud, _, _, _ = self.augment_lidar_point_cloud(point_cloud=lidar_point_cloud)
                clean_depth_map = self.lidar_to_depth_map(lidar_point_cloud)
                
                lidar_point_cloud, _, _, _ = self.augment_lidar_point_cloud(point_cloud=lidar_point_cloud)
                augmented_depth_map = self.lidar_to_depth_map(lidar_point_cloud)                    

            else:
                print(f'Lidar {lidar_file_path} Does not Exist!')
                exit(1)
                
            if label_file_path is not None:
                if os.path.exists(label_file_path):
                    class_labels, label_bboxes = self.read_label_file(label_file_path)                    

                    #left_image_arr : (375, 1242, 3)                 
                    transformed_dict = self.transform_sample(
                        left_image_arr, label_bboxes, class_labels
                    )

                    targets = self.prepare_targets_2d(
                        idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                    )

                    batch_data_items['targets'].append(
                        torch.tensor(targets, dtype=torch.float32)
                    )

                    # left_image = transformed_dict['image']
                    left_image_arr_augmented = self.transform_sample(left_image_arr_augmented)['image']
                    clean_depth_map = self.transform_sample(clean_depth_map)['image']
                    augmented_depth_map = self.transform_sample(augmented_depth_map)['image']

                else:
                    print(f'Label File Path not found!!')
                    exit(1)

            image_tensor = torch.from_numpy(left_image_arr_augmented).permute((2, 0, 1)) #(nc, h, w)
            clean_depth_map_tensor = torch.from_numpy(clean_depth_map).permute((2, 0, 1)) #(nc, h, w)  
            augmented_depth_map_tensor = torch.from_numpy(augmented_depth_map).permute((2, 0, 1)) #(nc, h, w)  

            batch_data_items['images'].append(image_tensor)
            batch_data_items['lidar_2d'].append(clean_depth_map_tensor)
            batch_data_items['augmented_lidar_2d'].append(augmented_depth_map_tensor)

        batch_data_items['images'] = torch.stack(
            batch_data_items['images'], dim=0
        ).float()

        batch_data_items['lidar_2d'] = torch.stack(
            batch_data_items['lidar_2d'], dim=0
        ).float()     

        batch_data_items['augmented_lidar_2d'] = torch.stack(
            batch_data_items['augmented_lidar_2d'], dim=0
        ).float()

        if batch_data_items['targets']:
            batch_data_items['targets'] = torch.concat(
                batch_data_items['targets'], dim=0
            )            

        return batch_data_items