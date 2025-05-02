import os, json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import seaborn as sns
import numpy as np

from .enums import Enums

class KiTTiDatasetEDA:
    def __init__(self, root_dir:str=None, split:str=None):
        
        self.root_dir = root_dir
        self.calib_dir = f'{self.root_dir}/calib'
        self.image_dir = f'{self.root_dir}/image_2'
        self.velodyne_dir = f'{self.root_dir}/velodyne'
        self.labels_dir = f'{self.root_dir}/label_2'
        
        self.split = split
        
    def read_labels(self):
        
        self.labels_data = defaultdict()
        
        for label_f in os.listdir(self.labels_dir):
            
            filepath = os.path.join(self.labels_dir, label_f)            
            class_labels, bboxes_2d, bboxes_3d, obj_levels = self.parse_label_file(filepath)
            
            self.labels_data[label_f] = {
                'class_labels':class_labels, 
                'bboxes_2d':bboxes_2d, 
                'bboxes_3d':bboxes_3d, 
                'obj_levels':obj_levels
            }

    def parse_label_file(self, filepath:str):

        class_labels = []
        bboxes_2d = []
        bboxes_3d = []
        
        obj_levels = []
        
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split() # Split the line into parts
                
                obj_type = parts[0]
                if Enums.mapping_dict and obj_type in Enums.mapping_dict:
                    obj_type = Enums.mapping_dict[obj_type]
                elif obj_type not in Enums.KiTTi_label2Id:
                    continue  # Skip invalid object types
                
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
                
                class_labels.append(obj_type)
                bboxes_2d.append([left, top, right, bottom])

                bboxes_3d.append([t[0], t[1], t[2], h, w, l, ry, dist_to_cam])
                
                obj_level = self.get_obj_level(
                    float(bottom) - float(top) + 1, 
                    float(parts[1]), 
                    int(parts[2])
                )
                
                obj_levels.append(obj_level)
                
        return class_labels, bboxes_2d, bboxes_3d, obj_levels
                
    def get_obj_level(self, height:float, truncation, occlusion):
                
        if height >= 40 and truncation <= 0.15 and occlusion <= 0:
            return 'Easy'  # Easy
        elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
            return 'Moderate'  # Moderate
        elif height >= 25 and truncation <= 0.5 and occlusion <= 3:
            return 'Hard'  # Hard

        # Extended categories (object is large enough)
        if height >= 25:
            if truncation > 0.5:
                return 'Truncated (High)'
            if occlusion > 2:
                return 'Occluded (Extreme)'
            return 'Other (Large but not matching known criteria)'

        # Small objects (height < 25)
        if truncation > 0.5 or occlusion > 2:
            return 'Small_and_Challenging'
        if truncation <= 0.5 and occlusion <= 2:
            return 'Small_Low_Trunc_Occl'

        print(f'UnKnown; height: {height:.2f}, truncation: {truncation}, occlusion: {occlusion}')
        return 'UnKnown'
    
    # 1. Label Distribution
    def plot_label_distribution(self, save_path):
        all_labels = []
        for v in self.labels_data.values():
            all_labels.extend(v['class_labels'])
        label_counts = Counter(all_labels)

        plt.figure(figsize=(10, 5))
        plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
        plt.title('Label Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)

    # 2. Box/Percentile Plot: Bbox Width/Height vs Label
    def plot_bbox_size_by_label(self, save_path):
        label_w, label_h = defaultdict(list), defaultdict(list)

        for v in self.labels_data.values():
            for label, box in zip(v['class_labels'], v['bboxes_2d']):
                width = box[2] - box[0]
                height = box[3] - box[1]
                label_w[label].append(width)
                label_h[label].append(height)

        labels = sorted(label_w.keys())
        data_w = [label_w[l] for l in labels]
        data_h = [label_h[l] for l in labels]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot(data_w, labels=labels, vert=True, showfliers=False, patch_artist=True)
        plt.title("BBox Width by Label")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.boxplot(data_h, labels=labels, vert=True, showfliers=False, patch_artist=True)
        plt.title("BBox Height by Label")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)

    # 3. Heatmap of bbox centers (val set assumed)
    def plot_bbox_center_heatmap(self, save_path, image_shape=(375, 1242)):
        center_map = np.zeros(image_shape)

        for v in self.labels_data.values():
            for box in v['bboxes_2d']:
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                if 0 <= y_center < image_shape[0] and 0 <= x_center < image_shape[1]:
                    center_map[y_center, x_center] += 1

        plt.figure(figsize=(10, 5))
        sns.heatmap(center_map, cmap='hot')
        plt.title("Heatmap of BBox Centers")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(save_path)

    # 4. Number of Easy, Moderate, Hard per class
    def plot_difficulty_distribution_per_class(self, save_path):
        class_difficulty = defaultdict(lambda: defaultdict(int))

        for v in self.labels_data.values():
            for label, diff in zip(v['class_labels'], v['obj_levels']):
                if diff in ['Easy', 'Moderate', 'Hard', 'Truncated (High)', 'Occluded (Extreme)']:
                    class_difficulty[label][diff] += 1

        classes = sorted(class_difficulty.keys())
        easy = [class_difficulty[c]['Easy'] for c in classes]
        moderate = [class_difficulty[c]['Moderate'] for c in classes]
        hard = [class_difficulty[c]['Hard'] for c in classes]
        trunc_high = [class_difficulty[c]['Truncated (High)'] for c in classes]
        occ_xtrem = [class_difficulty[c]['Occluded (Extreme)'] for c in classes]

        x = np.arange(len(classes))
        plt.figure(figsize=(12, 6))
        plt.bar(x - 0.2, easy, 0.2, label='Easy')
        plt.bar(x, moderate, 0.2, label='Moderate')
        plt.bar(x + 0.2, hard, 0.2, label='Hard')
        plt.bar(x + 0.2, trunc_high, 0.2, label='Truncated (High)')
        plt.bar(x + 0.2, occ_xtrem, 0.2, label='Occluded (Extreme)')
        plt.xticks(x, classes, rotation=45)
        plt.title("Difficulty Levels per Class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)

    # 5. Bbox Width & Height for Easy/Medium/Hard per class
    def plot_bbox_size_by_difficulty(self, save_path):
        diff_sizes = defaultdict(lambda: defaultdict(list))

        for v in self.labels_data.values():
            for label, box, diff in zip(v['class_labels'], v['bboxes_2d'], v['obj_levels']):

                width = box[2] - box[0]
                height = box[3] - box[1]
                diff_sizes[diff][label].append((width, height))

        for diff in diff_sizes:
            labels = sorted(diff_sizes[diff].keys())
            widths = [list(zip(*diff_sizes[diff][label]))[0] for label in labels]
            heights = [list(zip(*diff_sizes[diff][label]))[1] for label in labels]

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.boxplot(widths, labels=labels, showfliers=False, patch_artist=True)
            plt.title(f"{diff} - BBox Width")
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            plt.boxplot(heights, labels=labels, showfliers=False, patch_artist=True)
            plt.title(f"{diff} - BBox Height")
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_path = save_path.split('.')[0]
            plt.savefig(f'{save_path}_{diff}.png')
    
        
if __name__ == "__main__":
    
    eda = KiTTiDatasetEDA(
        '../data/KiTTi/training',
        'training'
    )
    
    eda.read_labels()
    
    OUTPUT_DIR = 'KiTTi-training-eda'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(f'{OUTPUT_DIR}/kitti_eda.json','w+') as f:
        json.dump(eda.labels_data, f)
        
    # Label Distribution
    eda.plot_label_distribution(f'{OUTPUT_DIR}/label_distribution.png')

    # BBox width/height vs label
    eda.plot_bbox_size_by_label(f'{OUTPUT_DIR}/bbox_size_per_label.png')

    # Heatmap
    eda.plot_bbox_center_heatmap(f'{OUTPUT_DIR}/bbox_center_heatmap.png')

    # Difficulty counts
    eda.plot_difficulty_distribution_per_class(f'{OUTPUT_DIR}/difficulty_distribution_per_class.png')

    # Width/Height per difficulty
    eda.plot_bbox_size_by_difficulty(f'{OUTPUT_DIR}/bbox_size_by_difficulty.png')    
        
        