import os, json, time
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import numpy as np
from terminaltables import AsciiTable
from collections import defaultdict

from model.mlsf_yolo import MLSFYolo
from model.mlsf_mobilenet import MLSFMobilenet
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiMLSFCollateFn, KittiMLSFMobilenet
from dataset_utils.nuscenes_2d_objectDetect import NuScenesObjectDetectDataset, NuScenesMLSFCollateFn
from dataset_utils.enums import Enums
from dataset_utils.kitti_eda import KiTTiDatasetEDA
from model.yolo_utils import xywh2xyxy, reshape_outputs, apply_sigmoid_activation, non_max_suppression, rescale_boxes, get_batch_statistics_eval, ap_per_class, compute_stats_per_difficulty
from test_yolo import draw_and_save_output_images

def compute_detection_stats_per_class(true_positives, false_positives, false_negatives, pred_labels, class_names):
    """
    Computes TP, FP, FN counts per class from detection results.

    Args:
        true_positives (np.ndarray): 1D array where each entry is 1 or 0 for each prediction.
        false_positives (np.ndarray): Same shape as true_positives, 1 where FP.
        false_negatives (np.ndarray): 1D array where each entry is 1 or 0 for each GT that wasn’t matched.
        pred_labels (np.ndarray): Class IDs for each prediction.
        class_names (List[str]): Ordered class name list aligned with label IDs.

    Returns:
        dict: {class_name: {'TP': ..., 'FP': ..., 'FN': ...}, ...}
    """
    stats_per_class = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    # Process TPs and FPs using predicted labels
    for tp, fp, label in zip(true_positives, false_positives, pred_labels):
        class_name = class_names[int(label)]
        if tp == 1:
            stats_per_class[class_name]['TP'] += 1
        elif fp == 1:
            stats_per_class[class_name]['FP'] += 1

    # Process FNs using GT labels (if available)
    # False negatives should contain class labels of missed GTs.
    missed_gt_classes = defaultdict(int)  # To track missed ground truth per class
    for label in false_negatives:
        missed_gt_classes[int(label)] += 1

    # Update FN stats per class
    for label, fn_count in missed_gt_classes.items():
        class_name = class_names[label]  # Assuming the label is already a valid class index
        stats_per_class[class_name]['FN'] += fn_count

    return stats_per_class

def print_eval_stats(metrics_output, class_names, output_dir:str,verbose=True):    
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP", "precision", "recall", "F1"]]
            for i, c in enumerate(ap_class):
                # ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                ap_table.append([
                    c, 
                    class_names[c],
                    f'{AP[i]:.5f}',
                    f'{precision[i]:.5f}',
                    f'{recall[i]:.5f}',
                    f'{f1[i]:.5f}'
                ])
            
            table_string = AsciiTable(ap_table).table
            
            print(f'---------- mAP per Class----------')
            print(f'{table_string}')
        
            print(f'---------- Total mAP {AP.mean():.5f} ----------')
            
            with open(f'{output_dir}/metrics.txt', 'w+') as f:
                f.write(table_string)
            
    else:
        print("---- mAP not measured (no detections found by model) ----") 

def create_dataloader_nuscenes(kwargs:dict, lidar_map_type:str="depth_map"):

    dataset = NuScenesObjectDetectDataset(
        table_blob_paths=kwargs['table_blob_paths'], 
        root_dir=kwargs['root_dir']
    )    
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=kwargs['batch_size'], 
        collate_fn=NuScenesMLSFCollateFn(
            image_resize=kwargs['image_resize'],                    
            lidar_map_type=lidar_map_type   
        )             
    )

    return dataloader    

def create_dataloader_kitti(test_dataset_kwargs:dict, lidar_map_type:str="depth_map"):
    dataset = Kitti2DObjectDetectDataset(
        lidar_dir=test_dataset_kwargs['lidar_dir'],
        calibration_dir=test_dataset_kwargs['calibration_dir'],
        left_image_dir=test_dataset_kwargs['left_image_dir'],
        right_image_dir=test_dataset_kwargs['right_image_dir'],
        labels_dir=test_dataset_kwargs['labels_dir']
    )
     
    if type(mlsf) == MLSFYolo:
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
    
    elif type(mlsf) == MLSFMobilenet:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=test_dataset_kwargs['batch_size'],
            collate_fn=KittiMLSFMobilenet(
                image_resize=test_dataset_kwargs['image_resize'],
                detection_head='yolo', 
                lidar_map_type=lidar_map_type
            ),
            shuffle=test_dataset_kwargs['shuffle']
        )
        
        return dataloader        

def load_mlsf_mobilenet(model_kwargs:dict, weights_path:str, image_resize):

    image_backbone_device = torch.device(model_kwargs['image_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    lidar_backbone_device = torch.device(model_kwargs['lidar_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    adaptive_fusion_device = torch.device(model_kwargs['adaptive_fusion_device']) if torch.cuda.is_available() else torch.device('cpu')    
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_kwargs['model_seed'])
        torch.cuda.manual_seed_all(model_kwargs['model_seed'])
        
    else:
        torch.manual_seed(model_kwargs['model_seed'])
    
    mlsf = MLSFMobilenet(
        image_resize=tuple(image_resize),
        detection_depths=model_kwargs['detection_depths'],        
        use_lidar_backbone=model_kwargs['use_lidar_backbone'],
        apply_adaptive_fusion=model_kwargs['apply_adaptive_fusion'],
        num_fusion_blocks=model_kwargs['num_fusion_blocks'], 
        fusion_type=model_kwargs['fusion_type'],         
        image_backbone_device=image_backbone_device, 
        lidar_backbone_device=lidar_backbone_device, 
        adaptive_fusion_device=adaptive_fusion_device,  
        # num_classes=len(Enums.KiTTi_label2Id),
        num_classes=len(Enums.nuscenes_label2Id)   
    )
    
    mlsf.load_state_dict(
        torch.load(weights_path)
    )    

    mlsf.image_backbone.to(mlsf.image_backbone_device)
    
    if mlsf.use_lidar_backbone:
        mlsf.lidar_backbone.to(mlsf.lidar_backbone_device)
    
    if mlsf.apply_adaptive_fusion:
        mlsf.adaptive_fusion_module.to(mlsf.adaptive_fusion_device)
    
    mlsf.detection_heads.to(mlsf.adaptive_fusion_device)
    
    return mlsf    

def load_mlsf_yolo(model_kwargs:dict, weights_path:str):
    
    image_backbone_device = torch.device(model_kwargs['image_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    lidar_backbone_device = torch.device(model_kwargs['lidar_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    adaptive_fusion_device = torch.device(model_kwargs['adaptive_fusion_device']) if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_kwargs['model_seed'])
        torch.cuda.manual_seed_all(model_kwargs['model_seed'])
        
    else:
        torch.manual_seed(model_kwargs['model_seed']) 
    
    mlsf = MLSFYolo(
        image_config_path=model_kwargs['image_cfg_file'], 
        lidar_config_path=model_kwargs['lidar_cfg_file'], 
        image_backbone_device=image_backbone_device, 
        lidar_backbone_device=lidar_backbone_device,
        adaptive_fusion_device=adaptive_fusion_device, 
        use_lidar_backbone=model_kwargs['use_lidar_backbone'],
        apply_adaptive_fusion=model_kwargs['apply_adaptive_fusion'], 
        num_fusion_blocks=model_kwargs['num_fusion_blocks'], 
        fusion_type=model_kwargs['fusion_type'], 
        weighted_fusion=model_kwargs['weighted_fusion'], 
        task_type=['obj_2d']
    )    
    
    if os.path.exists(weights_path):
        print(f'Weights Loaded: {weights_path}')
        mlsf.to(
            torch.device('cpu')
        )        

        mlsf.load_state_dict(
            torch.load(weights_path)
        )
        
        mlsf.image_backbone.to(mlsf.image_backbone_device)
        
        if mlsf.use_lidar_backbone:
            mlsf.lidar_backbone.to(mlsf.lidar_backbone_device)
        
        if mlsf.apply_adaptive_fusion:
            mlsf.adaptive_fusion_module.to(mlsf.adaptive_fusion_device)        
        
        return mlsf
    
    else:
        print(f'Weights path: {weights_path} does not exist')
        exit(1)
        
def test(mlsf:MLSFYolo, dataloader:torch.utils.data.DataLoader, image_resize, output_dir, 
        conf_thres=0.5, iou_thres=0.5, visualize:bool=False):
    
    mlsf.eval()
    test_iter = tqdm(dataloader)
    
    image_paths = []
    image_detections = []    
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    tp_per_level = defaultdict(int)
    
    time_per_batch = []
    
    if type(mlsf) == MLSFYolo:
        img_size = mlsf.image_backbone.hyperparams['height']
    elif type(mlsf) == MLSFMobilenet:
        img_size = mlsf.image_height
        
    
    for batch_idx, data_items in enumerate(test_iter):

        if not torch.is_tensor(data_items['targets']) or data_items['targets'].numel() == 0:
            continue
        
        with torch.no_grad():
            start = time.time()
            loss, _, outputs = mlsf(
                data_items['images'],
                data_items['lidar_2d'] if mlsf.use_lidar_backbone else None,
                data_items['targets']
            )
            
            end = time.time()
            
            time_per_batch.append(end - start)
        
        if type(dataloader.dataset) == Kitti2DObjectDetectDataset:
            obj_levels = []
            label_file_path = data_items['label_file_path']
            for idx, label_fp in enumerate(label_file_path):
                _, _, _, obj_level = KiTTiDatasetEDA().parse_label_file(label_fp)
                for level in obj_level:
                    obj_levels.append((idx, level))

        targets = data_items['targets'].cpu()
        labels += targets[:, 1] #[class_id]   
        
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        outputs = outputs['obj_2d']

        if type(mlsf) == MLSFYolo:
            anchor_grids = [yolo_layer.anchor_grid for yolo_layer in mlsf.image_backbone.yolo_layers]
        elif type(mlsf) == MLSFMobilenet:
            anchor_grids = [det_layer.anchor_grid for det_layer in mlsf.detection_heads.values()]
            
        outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=iou_thres)
                
        sample_metrics += get_batch_statistics_eval(outputs, targets, iou_threshold=0.5)
        
        if type(dataloader.dataset) == Kitti2DObjectDetectDataset:
            tp_stats_per_sample = compute_stats_per_difficulty(
                outputs, targets, obj_levels, iou_threshold=0.5
            )
            
            for stats in tp_stats_per_sample:
                for stat in stats:
                    _, difficulty = stat
                    tp_per_level[difficulty] += 1
                
        image_detections.extend(outputs)
        image_paths.extend(data_items['image_paths'])
        
        if (batch_idx + 1) % 10 == 0 and visualize:
            if image_detections:
                class_names = list(Enums.KiTTi_label2Id.keys())  
                draw_and_save_output_images(
                    image_detections, image_paths, 
                    image_resize[0],
                    f'{output_dir}', class_names
                )

                image_detections = []
                image_paths = []

    print(f'Detection Finished! Computing Metrics')
    
    true_positives, false_positives, false_negatives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]            

    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels) 
    
    class_names = list(Enums.KiTTi_label2Id.keys())    
    print_eval_stats(metrics_output, class_names, output_dir)
    
    stats = compute_detection_stats_per_class(true_positives, false_positives, false_negatives, pred_labels, class_names)
    
    with open(f'{output_dir}/detection_stats.json', 'w+') as f:
        json.dump(stats, f)
    
    with open(f'{output_dir}/time_per_batch.txt', 'w+') as f:
        f.write(f'Avg Time: {sum(time_per_batch)/len(time_per_batch)} Total time: {sum(time_per_batch)}')
    
    if type(dataloader.dataset) == Kitti2DObjectDetectDataset:
        with open(f'{output_dir}/tp_per_level.json', 'w+') as f:
            json.dump(tp_per_level, f)        
    

if __name__ == "__main__":
    
    test_kwargs = {
        "mlsf_yolo_kwargs":{
            "image_cfg_file":"config/yolov3-kitti-608.cfg",
            "lidar_cfg_file":"config/yolov3-yolo_reduced_classes_3D.cfg",
            # "image_cfg_file":"config/yolov3-nuscenes-608-numClasses-3.cfg",
            # "lidar_cfg_file":"config/yolov3-nuscenes-608-numClasses-3.cfg",            
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
        "mlsf_mobilenet_kwargs":{
            "cfg_file":"config/mobilenetv2.cfg",
            "image_channels":3, 
            "lidar_channels":3,
            "detection_depths":["InvertedResidual_6", "InvertedResidual_12", "InvertedResidual_17"],
            "image_backbone_device":"cuda:21",
            "lidar_backbone_device":"cuda:22",
            "adaptive_fusion_device":"cuda:23",
            "model_seed":101,
            "apply_adaptive_fusion":True,
            "use_lidar_backbone":True,
            "num_fusion_blocks":2, 
            "fusion_type":"attention"
        },        
        "kitti_validation_dataset_kwargs":{
            "lidar_dir":"data/KiTTi/validation/velodyne",
            "calibration_dir":"data/KiTTi/validation/calib",
            "left_image_dir":"data/KiTTi/validation/image_2",
            "right_image_dir":None,
            "labels_dir":"data/KiTTi/validation/label_2",
            "shuffle":False,
            "apply_augmentation":False, 
            "batch_size":12, 
            "image_resize":[608, 608], 
            "lidar_map_type":"bev_map"
        }, 
        "nuscenes_validation_dataset_kwargs":{
            "root_dir":"data/nuscenes",
            "shuffle":False,
            "batch_size":12,
            "table_blob_paths":[
                # "data/nuscenes/trainval04_blobs_US/tables.json", 
                # "data/nuscenes/trainval08_blobs_US/tables.json"
 
                "data/nuscenes/trainval10_blobs_US/tables.json"                
            ],
            "image_resize":[608, 608], 
            "perform_validation":False, 
            "lidar_map_type":"bev_map"                         
        },
    "output_dir":"eval_stats/MLSF-Mobilenet-608-RGB-BEV-Eval"
    }
    
    # yolo_model_path = "MLSF-YOLOv3-NuScenes-numClasses-3/best-model_obj_2d/best-model.pt"
    yolo_model_path = "results/mlsf_yolov3_ckpts_608/MLSF-YOLOv3-608-RGB-BEV/best-model_obj_2d/best-model.pt"
    # mobilenet_model_path = "results/mlsf_mobilenet/MLSF-Mobilenet-608-RGB/best-model_obj_2d/best-model.pt"
    mobilenet_model_path = "results/mlsf_mobilenet/MLSF-Mobilenet-608-RGB-BEV/best-model_obj_2d/best-model.pt"
    
    # mlsf = load_mlsf_yolo(
    #     test_kwargs['mlsf_yolo_kwargs'], yolo_model_path
    # )
    
    mlsf = load_mlsf_mobilenet(
        test_kwargs['mlsf_mobilenet_kwargs'], mobilenet_model_path, test_kwargs['kitti_validation_dataset_kwargs']['image_resize']
    )
    
    dataloader = create_dataloader_kitti(
        test_kwargs['kitti_validation_dataset_kwargs'], lidar_map_type=test_kwargs['kitti_validation_dataset_kwargs']['lidar_map_type']
    )

    # dataloader = create_dataloader_nuscenes(
    #     test_kwargs['nuscenes_validation_dataset_kwargs'], lidar_map_type=test_kwargs['nuscenes_validation_dataset_kwargs']['lidar_map_type']
    # )
    
    # if not os.path.exists(test_kwargs['output_dir']):
    #     os.makedirs(test_kwargs['output_dir'])
        
    # test(
    #     mlsf=mlsf,
    #     dataloader=dataloader, 
    #     image_resize=test_kwargs["kitti_validation_dataset_kwargs"]['image_resize'],
    #     output_dir=test_kwargs['output_dir'],
    #     conf_thres=0.25,
    #     iou_thres=0.45
    # )        
    
    for conf in [0.25, 0.4, 0.5, 0.65]:
        output_dir = f"{test_kwargs['output_dir']}/Conf_thresh_{conf}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        test(
            mlsf=mlsf,
            dataloader=dataloader, 
            image_resize=test_kwargs["kitti_validation_dataset_kwargs"]['image_resize'],
            output_dir=output_dir,
            conf_thres=conf,
            iou_thres=0.45,
            visualize=False
        )
        
        exit(1)
    