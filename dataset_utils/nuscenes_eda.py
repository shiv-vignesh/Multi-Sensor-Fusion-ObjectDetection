import os, json 
from collections import defaultdict, Counter
from typing import Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from enums import Enums

class NuScenesEda:
    
    def __init__(self, table_blobs_json:list, split:str):
        
        self.table_blobs_json = table_blobs_json
        self.split = split
        
        self.labels_data = defaultdict()
        
        for table_json in self.table_blobs_json:
            self.load_tables(table_json)
        
    def load_tables(self, table_json):
        
        table = json.load(open(table_json))
        
        for sample_token in table:
            labels_2d_cam_front = table[sample_token]['labels_2d_cam_front']
            class_labels, label_bboxes_2d = self.preprocess_labels(labels_2d_cam_front)
            
            self.labels_data[sample_token] = {
                'class_labels':class_labels, 
                'label_bboxes_2d':label_bboxes_2d
            }
            
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
                # class_id = Enums.nuscenes_label2Id[class_label]
                
                class_labels.append(class_label)
                bboxes_2d.append([left, top, right, bottom])

        return class_labels, bboxes_2d
    
    def plot_label_distribution(self, save_path):
        counter = Counter()
        for entry in self.labels_data.values():
            counter.update(entry['class_labels'])

        labels, counts = zip(*sorted(counter.items()))        

        plt.figure(figsize=(12, 6))
        sns.barplot(x=labels, y=counts, palette="tab10")
        plt.title("Label Distribution")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(save_path)

    def plot_bbox_size_by_label(self, save_path):
        label_bbox = defaultdict(list)

        for entry in self.labels_data.values():
            for class_id, bbox in zip(entry['class_labels'], entry['label_bboxes_2d']):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                label_bbox[class_id].append((w, h))

        labels = sorted(label_bbox.keys())
        widths = [list(zip(*label_bbox[l]))[0] for l in labels]
        heights = [list(zip(*label_bbox[l]))[1] for l in labels]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        box1 = plt.boxplot(widths, labels=labels, patch_artist=True, showfliers=False)
        for patch in box1['boxes']:
            patch.set_facecolor('#99ccff')
        plt.title("BBox Width by Label")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        box2 = plt.boxplot(heights, labels=labels, patch_artist=True, showfliers=False)
        for patch in box2['boxes']:
            patch.set_facecolor('#ffcc99')
        plt.title("BBox Height by Label")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)

    def plot_bbox_center_heatmap(self, save_path):
        centers_x, centers_y = [], []

        for entry in self.labels_data.values():
            for bbox in entry['label_bboxes_2d']:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                centers_x.append(cx)
                centers_y.append(cy)

        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=centers_x, y=centers_y, cmap="Reds", fill=True, thresh=0.05)
        plt.title("BBox Center Heatmap")
        plt.xlabel("X (image width)")
        plt.ylabel("Y (image height)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path)

    def get_difficulty(self, height):
        if height < 40:
            return 'Small'
        elif height < 100:
            return 'Medium'
        else:
            return 'Large'

    def plot_difficulty_distribution_per_class(self, save_path):
        class_difficulty = defaultdict(lambda: Counter())

        for entry in self.labels_data.values():
            for class_id, bbox in zip(entry['class_labels'], entry['label_bboxes_2d']):
                h = bbox[3] - bbox[1]
                difficulty = self.get_difficulty(h)
                class_difficulty[class_id][difficulty] += 1

        label_names = [l for l in sorted(class_difficulty)]
        categories = ['Small', 'Medium', 'Large']
        data = np.array([
            [class_difficulty[l][c] for c in categories]
            for l in sorted(class_difficulty)
        ])

        plt.figure(figsize=(12, 6))
        bottom = np.zeros(len(label_names))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        for i, cat in enumerate(categories):
            plt.bar(label_names, data[:, i], bottom=bottom, label=cat, color=colors[i])
            bottom += data[:, i]

        plt.title("Difficulty Distribution per Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.legend(title="Difficulty")
        plt.tight_layout()
        plt.savefig(save_path)

    def plot_bbox_size_by_difficulty(self, save_path):
        difficulty_data = defaultdict(list)

        for entry in self.labels_data.values():
            for bbox in entry['label_bboxes_2d']:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                difficulty = self.get_difficulty(h)
                difficulty_data[difficulty].append((w, h))

        difficulties = ['Small', 'Medium', 'Large']
        widths = [list(zip(*difficulty_data[d]))[0] for d in difficulties]
        heights = [list(zip(*difficulty_data[d]))[1] for d in difficulties]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        box1 = plt.boxplot(widths, labels=difficulties, patch_artist=True, showfliers=False)
        for patch in box1['boxes']:
            patch.set_facecolor('#b3cde0')
        plt.title("BBox Width by Difficulty")

        plt.subplot(1, 2, 2)
        box2 = plt.boxplot(heights, labels=difficulties, patch_artist=True, showfliers=False)
        for patch in box2['boxes']:
            patch.set_facecolor('#fbb4ae')
        plt.title("BBox Height by Difficulty")

        plt.tight_layout()
        plt.savefig(save_path)
    
if __name__ == "__main__":
    
    train_blobs_path = [
                    "../data/nuscenes/trainval03_blobs_US/tables.json",                     
                    "../data/nuscenes/trainval05_blobs_US/tables.json", 
                    "../data/nuscenes/trainval06_blobs_US/tables.json",
                    "../data/nuscenes/trainval07_blobs_US/tables.json"

                ]
    
    validation_blobs_path = [
                    "../data/nuscenes/trainval04_blobs_US/tables.json", 
                    "../data/nuscenes/trainval08_blobs_US/tables.json"
                ]

    OUTPUT_DIR = 'nuscenes-validation-eda'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    eda = NuScenesEda(validation_blobs_path, split='val')
    
    eda.plot_label_distribution(f'{OUTPUT_DIR}/label_distribution.png')
    eda.plot_bbox_size_by_label(f'{OUTPUT_DIR}/bbox_size_per_label.png')
    eda.plot_bbox_center_heatmap(f'{OUTPUT_DIR}/bbox_center_heatmap.png')
    eda.plot_difficulty_distribution_per_class(f'{OUTPUT_DIR}/difficulty_per_class.png')
    eda.plot_bbox_size_by_difficulty(f'{OUTPUT_DIR}/bbox_size_per_difficulty.png')
    