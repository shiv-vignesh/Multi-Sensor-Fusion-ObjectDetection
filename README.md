# Multi-Modal 2D and 3D Object Detection

This project implements a multi-modal 2D object detection system that fuses LiDAR point cloud data with camera images using the KITTI and NuScenes dataset.

Developed by Shiv Vignesh. 

## Highlights

- Twin Backbone Fusion
  - 🗺️ Converts to 2D (Depth, BEV)​
  - ⚡ Parallel feature extraction​, same model architecture for image and lidar stream.

### Architecture Comparison: YOLOv3 vs MobileNetV2 (as Encoder)

| **Feature**            | **YOLOv3 (FPN)**                            | **MobileNetV2 (Encoder)**                          |
|------------------------|--------------------------------------------|---------------------------------------------------|
| **Feature Pyramid**    | ✅ Built-in, strong for multi-scale         | ❌ Must be manually constructed                    |
| **Speed**              | ❌ Slower                                   | ✅ Faster                                          |
| **Fusion Flexibility** | ❌ Rigid (13/26/52)                         | ✅ Flexible (depth-level control)                 |
| **Accuracy (default)** | ✅ Higher for detection tasks               | ⚠️ Lower unless tuned                             |
| **Anchor Adaptability**| ✅ Anchors matched to feature maps          | ⚠️ Requires tuning for custom depths              |


- Attention Based Fusion
  - 🎯 Cross-attention + adaptive weighting​
  - 🔍 Prioritizes important features dynamically​

### Code Contributions
- Part of the codebase for YOLOv3 implementation is adapted from [PyTorch-YOLOv3 by Erik Lindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master).
  - Please also refer to the following paper:
    ```
    @article{redmon2018yolov3,
      title={YOLOv3: An Incremental Improvement},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1804.02767},
      year={2018}
    }
    ```

### Dataset
- This project utilized the [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) for training and evaluation purposes. 
  ```
  Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). 
  Vision meets robotics: The kitti dataset. 
  The International Journal of Robotics Research, 32(11), 1231-1237.
  ```

## Data Preprocessing
![data_preprocessing_pipeline](samples/datapreproessing_steps_visualize.png)

<!-- ## Yolov3 + Pointnet : Spatial Transformation Fusion

### Model Pipeline
![Spatial-IL Fusion Pipeline](samples/Proposed-Methodology-Complete.png) -->

### Detection Outputs
![Spatial-IL Fusion Pipeline Detection](samples/000152_detection.png)
![Spatial-IL Fusion Pipeline Detection](samples/000181_detection.png)
![Spatial-IL Fusion Pipeline Detection](samples/000618_detection.png)

## Twin Backbone Fusion

### Model Pipeline
![Twin Backbone Fusion](samples/twinbackbone_model_fusion.png)

### Detection Outputs
<!-- ![Twin Backbone Pipeline Detection](results/MLSF-YOLO-Attention-FocalLoss/best-model/detections/001872.png)
![Twin Backbone Pipeline Detection](results/MLSF-YOLO-Attention-FocalLoss/best-model/detections/000073.png)
![Twin Backbone Pipeline Detection](results/MLSF-YOLO-Attention-FocalLoss/best-model/detections/002892.png) -->

![Spatial-IL Fusion Pipeline Detection](samples/000152_detection.png)
![Spatial-IL Fusion Pipeline Detection](samples/000181_detection.png)
![Spatial-IL Fusion Pipeline Detection](samples/000618_detection.png)

## Adaptive Fusion Block

<div style="display: flex; justify-content: space-around;">
    <div style="flex: 0 0 48%;">
        <img src="samples/Simple-Adaptive-Fusion.png" alt="Simple-Adaptive-Fusion" style="width: 100%;"/>
        <p> <b>Simple-Adaptive-Cross-Fusion</b>
Feature Concatenation: Image and LiDAR features are directly concatenated along the channel dimension.​ 
        
Shared Feature Extraction: A 3×3 convolutional layer learns shared representations from both modalities. 

The extracted features are split back into separate image and LiDAR feature maps.​ 

1×1 convolutions refine the features independently for each modality, adapting them to their respective networks.​</p>
    </div>
    <div style="flex: 0 0 48%;">
        <img src="samples/Attention Based Cross-Fusion​.png" alt="Attention Based Cross Fusion" style="width: 100%;"/>
        <p><b>Attention Based Cross-Fusion</b>

Dimension Reduction: Compresses image and LiDAR features via 1×1 convolutions for efficiency.​

Cross-Attention: Uses Query, Key, and Value to compute attention weights, highlighting important features.​

Adaptive Fusion: Reweights and fuses attended features dynamically.​

Gating Mechanism (if enabled): Learns optimal weighting of image and LiDAR features based on context.</p>
    </div>
</div>

---

## Usage

1. Prepare the KITTI dataset:

    - Download the KITTI object detection dataset
    - Organize the data in the following structure:

            data/
                ├── training/
                │   └── calib/
                │   └── image_2/
                │   └── label_2/
                │   └── velodyne/
                ├── validation/
                │   └── calib/
                │   └── image_2/
                │   └── label_2/
                │   └── velodyne/                



<!-- 1. **YOLO-PointNet Fusion**  
   📄 Config: [`config/yolo_pointnet_fusion_trainer.json`](config/yolo_pointnet_fusion_trainer.json) -->

1. **Twin-Backbone MLSF-YOLOv3 Fusion**  
   📄 Config: [`config/twin_backbone_trainer.json`](config/mlsf_trainer.json)

1. **Twin-Backbone MLSF-YOLOv3 Fusion**  
   📄 Config: [`config/twin_backbone_trainer.json`](config/mlsf_trainer.json)


Both models support feature-level fusion, independent backbone training, and robustness augmentations.

---

## 🔧 Common Configuration Blocks

The following sections are common to both model configurations.

### `dataset_kwargs`

Controls dataset paths and preprocessing:

```json
"dataset_kwargs": {
  "image_resize": [640, 640],
  "perform_validation": false,
  "kitti_trainer_dataset_kwargs": {
    "lidar_dir": "data/KiTTi/training/velodyne",
    "left_image_dir": "data/KiTTi/training/image_2",
    "labels_dir": "data/KiTTi/training/label_2",
    "shuffle": true,
    "apply_augmentation": false,
    "batch_size": 8
  },
  "kitti_validation_dataset_kwargs": {
    "lidar_dir": "data/KiTTi/validation/velodyne",
    "left_image_dir": "data/KiTTi/validation/image_2",
    "labels_dir": "data/KiTTi/validation/label_2",
    "shuffle": false,
    "apply_augmentation": false,
    "batch_size": 12
  }
}
```

---

### `trainer_kwargs`

Manages the training loop and evaluation strategy:

```json
"trainer_kwargs": {
  "output_dir": "output_dir_name",
  "is_training": true,
  "epochs": 50,
  "first_val_epoch": 0,
  "monitor_train": true,
  "monitor_val": true,
  "gradient_clipping": 1.0,
  "gradient_accumulation_steps": 4,
  "checkpoint_idx": 5,
  "robustness_augmentations": ["SaltPapperNoise", "pixelate"]
}
```

<!-- ---

## 🔍 YOLO-PointNet Fusion  
📄 Config: [`config/yolo_pointnet_fusion_trainer.json`](config/yolo_pointnet_fusion_trainer.json)

This model fuses YOLO-based image features with PointNet features from LiDAR.

### ➕ `pointnet_kwargs`

```json
"pointnet_kwargs": {
  "num_points": 75000,
  "num_global_feats": 1024
}
```

### 🔁 `adaptive_fusion_kwargs`

```json
"adaptive_fusion_kwargs": {
  "fusion_type": "residual",
  "transform_image_features": false,
  "alpha": 1.0
}
```

### ⚙️ `optimizer_kwargs`

```json
"optimizer_kwargs": {
  "type": "AdamW",
  "train_yolo_backbone": true,
  "train_yolo_detection": true,
  "train_pointnet": true,
  "train_fusion_layers": true,
  "pointnet_lr": 3e-5,
  "pointnet_momentum": 0.9,
  "pointnet_decay": 1e-4,
  "fusion_lr": 5e-3,
  "fusion_momentum": 0.9,
  "fusion_decay": 1e-4
}
```

--- -->

## 🚀 Twin-Backbone MLSF-YOLOv8 Fusion  
📄 Config: [`config/twin_backbone_trainer.json`](config/twin_backbone_trainer.json)

This model uses separate image and LiDAR backbones with attention-based fusion.

### 🧠 `model_kwargs`

```json
"model_kwargs": {
  "model_type": "mlsf_yolov8",
  "mlsf_yolo_kwargs": {
    "cfg_file": "config/yolov3-yolo_reduced_classes.cfg",
    "yolov8_weights_path": "yolov8l.pt",
    "image_channels": 3,
    "lidar_channels": 3,
    "image_backbone_device": "cuda:16",
    "lidar_backbone_device": "cuda:17",
    "adaptive_fusion_device": "cuda:19",
    "apply_adaptive_fusion": true,
    "fusion_type": "attention",
    "num_fusion_blocks": 2,
    "weighted_fusion": false,
    "model_seed": 101
  }
}

```
## 📝 Note on `config/yolov3-yolo_reduced_classes.cfg` and YOLOv3 Architecture

The file `config/yolov3-yolo_reduced_classes.cfg` defines the configuration for a **YOLOv3** model with a reduced number of classes that is used by both the methods. YOLOv3 implementation is adapted from [PyTorch-YOLOv3 by Erik Lindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master). 


> Optionally supports `mlsf_ssd_kwargs` for MobileNetV2-SSD-based fusion.

### ⚙️ `optimizer_kwargs`

```json
"optimizer_kwargs": {
  "type": "AdamW",
  "image_backbone_lr": 3e-3,
  "lidar_backbone_lr": 3e-3,
  "adaptive_fusion_lr": 3e-3,
  "ssd_head_lr": 5e-3,
  "momentum": 0.9,
  "weight_decay": 0.9,
  "tune_lidar_backbone": true,
  "tune_image_backbone": true
}
```

---

## 🧪 Robustness Options

Both models support robustness configurations:

- `"robustness_augmentations"`: e.g., `["SaltPapperNoise", "pixelate"]`
- `"modality_dropout"` / `"modality_corrupt"`: Enable modality dropout/corruption
- `"p_modality_dropout"`: Probability of modality dropout during training

---

## 🖥 Device Control

Assign specific GPUs to each component for efficient multi-GPU training:

- `yolo_device_id`, `pointnet_device_id`
- `image_backbone_device`, `lidar_backbone_device`, `adaptive_fusion_device`

---

## 🔁 Resuming Training

- Set `"checkpoint_idx"` to resume from a specific checkpoint.
- Outputs are saved to `"output_dir"` with logs and models.

---

## 🚀 Run Training

```bash
# YOLO-PointNet
python train_adaptive_fusion.pyjson

# Twin-Backbone Fusion
python train_mlsf.py
```

---

## 📎 Notes

- Make sure your dataset follows the KITTI format.
- Ensure all paths and device IDs are valid before running.
- Modify `image_resize` consistently across model and dataset configs.

---

Happy fusing! 🚘📦📡



