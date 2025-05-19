from typing import List, Tuple, Union
from collections import defaultdict, OrderedDict, Counter
import numpy as np
from shapely.geometry import MultiPoint, box
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import os, json
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm

def draw_2d_boxes(image_path: str, boxes: List[OrderedDict]):
    img = cv2.imread(image_path)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box['bbox_corners'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(img, box['category_name'].split('.')[-1], (x1, y1 - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0), thickness=1)

    return img

def compute_unique_labels(attribute_counts:Counter, save_path:str):

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(attribute_counts.keys(), attribute_counts.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Unique Object Attributes in Sample')
    plt.xlabel('Attribute')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig(save_path)

class NuScenesParser:

    def __init__(self, root_dir:str, labels_dir:str, blobs_dir:str=None):
        
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.blobs_dir = blobs_dir
        
        self.use_blobs_path = True if self.blobs_dir is not None else False
        
        self.ego_pose_path = os.path.join(root_dir, labels_dir, 'ego_pose.json')
        self.annotation_path = os.path.join(root_dir, labels_dir, 'sample_annotation.json')
        self.attribute_path = os.path.join(root_dir, labels_dir, 'attribute.json')
        
        self.table = self.build_sample_table()        
        
        self.add_ego_pose()
        self.add_annotations()

    def build_sample_table(self):

        table = defaultdict(dict)

        sample_data_path = os.path.join(self.root_dir, self.labels_dir, 'sample_data.json')
        calibrated_sensor_path = os.path.join(self.root_dir, self.labels_dir, 'calibrated_sensor.json')

        sample_data = json.load(open(sample_data_path))
        calibrated_data = {cs['token']: cs for cs in json.load(open(calibrated_sensor_path))}

        for entry in tqdm(sample_data):

            if 'sweeps' in entry['filename']:
                continue

            sample_token = entry['sample_token']
            filename = entry['filename']
            modality = filename.split('/')[1].lower()
            
            if modality in ['cam_front', 'lidar_top']:
                
                if self.use_blobs_path:
                    table[sample_token][f'{modality}_fp'] = os.path.join(self.blobs_dir, filename)
                else:
                    table[sample_token][f'{modality}_fp'] = os.path.join(self.root_dir, filename)
                
                table[sample_token][f'calibrated_sensor_token_{modality}'] = entry['calibrated_sensor_token']
                table[sample_token][f'calibrated_sensor_{modality}'] = calibrated_data.get(entry['calibrated_sensor_token'], {})
                table[sample_token][f'ego_pose_token_{modality}'] = entry['ego_pose_token']

        return table
    
    def add_ego_pose(self):
        ego_data = {e['token']: e for e in json.load(open(self.ego_pose_path))}
        for sample_token, v in self.table.items():
            for sensor in ['cam_front', 'lidar_top']:
                token = v.get(f'ego_pose_token_{sensor}')
                if token:
                    v[f'ego_pose_{sensor}'] = ego_data[token]

    def add_annotations(self):
        annotations = json.load(open(self.annotation_path))
        attributes = {a['token']: a['name'] for a in json.load(open(self.attribute_path))}

        for ann in annotations:
            sample_token = ann['sample_token']
            box = {
                'attribute': attributes.get(ann['attribute_tokens'][0], '') if ann['attribute_tokens'] else '',
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
            }
            self.table[sample_token].setdefault('labels', []).append(box)

        return self.table

class ConvertToKiTTi:
    
    def __init__(self, table:dict, root_dir:str, blobs_dir:str, 
                version:str, visibilities:list=['', '1', '2', '3', '4']):        
        
        self.table = table
        self.root_dir = root_dir
        
        self.blobs_dir = blobs_dir
        self.use_blobs_path = True if self.blobs_dir is not None else False
        
        self.nuscenes = NuScenes(version=version, dataroot=root_dir, verbose=False)
        self.visibilities = visibilities
        self.inspect = True

        if not os.path.exists(f'visual_inspections_2'):
            os.makedirs(f'visual_inspections_2')
            
        self.visual_dir = f'visual_inspections_2'
        self.final_table = {}
        
    def project_bbox(self):
        
        """
        token	Unique ID of the data record	Get or link a specific sample_data
        sample_token	Belongs to a timestamp group	Group all sensors together at a single timestep    
        """

        sample_data_camera_tokens = []    
        for s in tqdm(self.nuscenes.sample_data):
            if s['sensor_modality'] == 'camera' and s['channel'] == 'CAM_FRONT' and s['is_key_frame']:
                if self.use_blobs_path:
                    filename = f'{self.blobs_dir}/{s["filename"]}'
                else:
                    filename = f'{self.root_dir}/{s["filename"]}'

                if os.path.exists(filename):                
                    sample_data_camera_tokens.append({
                        'token':s['token'],
                        'filename':filename,
                        'sample_token':s['sample_token']
                    })

        attributes = []
        for idx, data in tqdm(enumerate(sample_data_camera_tokens)):
            box_2d = self.get_2d_boxes_from_camera(
                nusc=self.nuscenes, 
                sample_data_token=data['token'], 
                filename=data['filename'],
                visibilities=self.visibilities
            )

            data['box_2d_cam_front'] = box_2d
            sample_data_camera_tokens[idx] = data
            
            sample_token = data['sample_token']

            del self.table[sample_token]['labels']
            
            self.table[sample_token]['labels_2d_cam_front'] = box_2d
            attributes.extend([box['category_name'] for box in box_2d])
            
            self.final_table[sample_token] = self.table[sample_token]
            
            if idx % 500 == 0:
                img = draw_2d_boxes(
                    data['filename'], 
                    box_2d
                )

                cv2.imwrite(f'{self.visual_dir}/sample_visual_{idx}.png', img)
                
        attribute_counts = Counter(attributes)
        
        compute_unique_labels(attribute_counts, f'{self.blobs_dir}/label_distribution.png')
        
        with open(f'{self.blobs_dir}/unique_counts.json','w+') as f:
            json.dump(attribute_counts, f)

    def post_process_coords(self, corner_coords: List,
                            imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None

    def generate_record(self, ann_rec: dict,
                        x1: float,
                        y1: float,
                        x2: float,
                        y2: float,
                        sample_data_token: str,
                        filename: str) -> OrderedDict:
        repro_rec = OrderedDict()
        repro_rec['sample_data_token'] = sample_data_token

        relevant_keys = [
            'attribute_tokens',
            'category_name',
            'instance_token',
            'next',
            'num_lidar_pts',
            'num_radar_pts',
            'prev',
            'sample_annotation_token',
            'sample_data_token',
            'visibility_token',
        ]

        for key in relevant_keys:
            if key in ann_rec:
                repro_rec[key] = ann_rec[key]

        repro_rec['bbox_corners'] = [x1, y1, x2, y2]
        repro_rec['filename'] = filename

        return repro_rec

    def get_2d_boxes_from_camera(self, nusc: NuScenes,
                                sample_data_token: str,
                                filename: str,
                                visibilities: List[str]) -> List[OrderedDict]:
        
        sd_rec = nusc.get('sample_data', sample_data_token)
        assert sd_rec['sensor_modality'] == 'camera' and sd_rec['is_key_frame'], 'Expected CAM keyframe.'

        s_rec = nusc.get('sample', sd_rec['sample_token'])
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

        ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
        ann_recs = [a for a in ann_recs if a['visibility_token'] in visibilities]

        repro_recs = []
        for ann in ann_recs:
            ann['sample_annotation_token'] = ann['token']
            ann['sample_data_token'] = sample_data_token

            box = nusc.get_box(ann['token'])

            # Global → ego
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # Ego → cam
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            corners_3d = box.corners()
            in_front = corners_3d[2, :] > 0
            corners_3d = corners_3d[:, in_front]

            if corners_3d.shape[1] < 1:
                continue

            corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
            final_coords = self.post_process_coords(corner_coords)

            if final_coords is None:
                continue

            x1, y1, x2, y2 = final_coords
            repro_rec = self.generate_record(ann, x1, y1, x2, y2, 
                                            sample_data_token, filename)
            repro_recs.append(repro_rec)

        return repro_recs

if __name__ == "__main__":
    
    root_dir = '../data/nuscenes'
    # labels_dir = 'v1.0-mini'
    blobs_dir = [
                # '../data/nuscenes/trainval03_blobs_US', 
                # '../data/nuscenes/trainval04_blobs_US', 
                # '../data/nuscenes/trainval05_blobs_US', 
                '../data/nuscenes/trainval10_blobs_US', 
                # '../data/nuscenes/trainval07_blobs_US', 
                # '../data/nuscenes/trainval08_blobs_US'
                ]

    labels_dir = 'v1.0-trainval'    
    
    for blob_dir in blobs_dir:    
        parser = NuScenesParser(root_dir, 
                                labels_dir, 
                                blob_dir)
        
        kitti_convert = ConvertToKiTTi(
            parser.table, root_dir, 
            blob_dir, labels_dir, visibilities=['4']
        )
        
        kitti_convert.project_bbox()
        
        with open(f'{blob_dir}/tables.json', 'w+') as f:
            json.dump(kitti_convert.final_table, f)