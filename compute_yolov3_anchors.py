import os, json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans

def read_nuscenes_labels(blob_labels_paths:list):
    
    objects_width, objects_height = [], []
    for label_path in tqdm(blob_labels_paths):
        blob_labels = json.load(open(label_path))
        
        for sample_token in blob_labels:
            sample_data = blob_labels[sample_token]

            for box_2d_data in sample_data['labels_2d_cam_front']:
                parts = box_2d_data['bbox_corners']
                left = float(parts[0])  # left
                top = float(parts[1])  # top
                right = float(parts[2])  # right
                bottom = float(parts[3])  # bottom

                width = abs(right - left)
                height = abs(bottom - top)
                
                objects_width.append(width)
                objects_height.append(height)
                
    width_arr = np.array(objects_width)
    height_arr = np.array(objects_height)
    
    return width_arr, height_arr

def read_kitti_labels(label_file_path:str):
    
    w, h = [], []
    with open(label_file_path, 'r') as file:
        for line in file:                
            parts = line.strip().split() # Split the line into parts
            
            left = float(parts[4])  # left
            top = float(parts[5])  # top
            right = float(parts[6])  # right
            bottom = float(parts[7])  # bottom  
            
            width = abs(right - left)
            height = abs(bottom - top)
            
            w.append(width)
            h.append(height)
            
    return w, h

def process_labels(labels_dir:str):
    
    objects_width, objects_height = [], []
    for label_file in os.listdir(labels_dir):
        w, h = read_kitti_labels(f'{labels_dir}/{label_file}')
        objects_height.extend(w)
        objects_width.extend(h)
        
    return objects_width, objects_height

def kitti_box_wh(train_labels_dir:str, val_labels_dir:str=None):
    objects_width, objects_height = process_labels(train_labels_dir)
    
    if val_labels_dir is not None:
        val_objects_width, val_objects_height = process_labels(val_labels_dir)
        objects_width.extend(val_objects_width)
        objects_height.extend(val_objects_height)
        
    width_arr = np.array(objects_width)
    height_arr = np.array(objects_height)
    
    return width_arr, height_arr

def compute_anchors(width_arr:np.array, height_arr:np.array):
    
    x = np.stack([width_arr, height_arr], axis=1) #(len, 2)
    
    # K-Means clustering to generate 9 anchor boxes; 3 per yolo_layer
    kmeans = KMeans(n_clusters=9)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)
        
    # Compute average width/height for each cluster
    anchors = []
    for i in range(9):
        anchors.append(np.mean(x[y_kmeans == i], axis=0))
    anchors = np.array(anchors)

    # Rescale anchors to YOLO input size
    scaled_anchors = anchors.copy()
    scaled_anchors[:, 0] = (anchors[:, 0] / IMG_WIDTH) * YOLO_INPUT_SIZE
    scaled_anchors[:, 1] = (anchors[:, 1] / IMG_HEIGHT) * YOLO_INPUT_SIZE
    scaled_anchors = np.rint(scaled_anchors)
    
    # Plot anchor boxes
    fig, ax = plt.subplots()
    for i in range(9):
        rect = plt.Rectangle((YOLO_INPUT_SIZE/2 - scaled_anchors[i,0]/2, YOLO_INPUT_SIZE/2 - scaled_anchors[i,1]/2),
                            scaled_anchors[i,0], scaled_anchors[i,1],
                            edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    ax.set_aspect(1.0)
    plt.xlim([0, YOLO_INPUT_SIZE])
    plt.ylim([0, YOLO_INPUT_SIZE])
    plt.title("YOLOv3 Anchor Boxes")
    plt.savefig(PLT_SAVE_PATH)

    # Save anchors
    scaled_anchors = scaled_anchors.astype(int)
    scaled_anchors = scaled_anchors[np.argsort(scaled_anchors[:, 0])]
    print("Your custom anchor boxes are:\n", scaled_anchors)

    with open(ANCHORS_TXT, "w+") as f:
        f.write(str(scaled_anchors.tolist()))    
        

if __name__ == "__main__":
    
    # # ===== KITTI =====
    # IMG_WIDTH = 1242
    # IMG_HEIGHT = 375
    # YOLO_INPUT_SIZE = 224  
    
    # PLT_SAVE_PATH = f"data/kitti_yolov3_anchors_{YOLO_INPUT_SIZE}.png"  
    # ANCHORS_TXT = f'data/YOLOV3_KITTI_Anchors_{YOLO_INPUT_SIZE}.txt'
    
    # width_arr, height_arr = kitti_box_wh(
    #     "data/KiTTi/training/label_2",
    #     "data/KiTTi/validation/label_2"
    # )
    
    # ==== NuScenes ====
    
    IMG_WIDTH = 1600
    IMG_HEIGHT = 900
    YOLO_INPUT_SIZE = 608  
    
    PLT_SAVE_PATH = f"data/nuscenes_yolov3_anchors_{YOLO_INPUT_SIZE}.png"  
    ANCHORS_TXT = f'data/YOLOV3_nuscenes_Anchors_{YOLO_INPUT_SIZE}.txt'
    
    width_arr, height_arr = read_nuscenes_labels(
        [
            'data/nuscenes/trainval03_blobs_US/tables.json', 
            'data/nuscenes/trainval04_blobs_US/tables.json', 
            'data/nuscenes/trainval05_blobs_US/tables.json', 
            'data/nuscenes/trainval06_blobs_US/tables.json',
            'data/nuscenes/trainval07_blobs_US/tables.json',
            'data/nuscenes/trainval08_blobs_US/tables.json',
            'data/nuscenes/trainval10_blobs_US/tables.json',
        ]
    )    
    
    compute_anchors(
        width_arr, height_arr
    )