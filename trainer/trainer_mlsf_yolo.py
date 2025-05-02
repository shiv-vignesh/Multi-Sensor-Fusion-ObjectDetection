import os, math, time, random
from tqdm import tqdm
import torch, torchvision
from typing import Iterable, Union
from collections import defaultdict
from terminaltables import AsciiTable
import numpy as np

from .logger import Logger
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiMLSFCollateFn, KitiiMLSFCollateAugment, KittiMLSFMobilenet
from dataset_utils.nuscenes_2d_objectDetect import NuScenesObjectDetectDataset, NuScenesMLSFCollateFn
from dataset_utils.enums import Enums
from model.mlsf_yolo import MLSFYolo
from model.mlsf_mobilenet import MLSFMobilenet
from model.mlsf_yolov8 import MLSFYolov8
# from model.mlsf_utils import decode_boxes, compute_precision_recall_f1
from model.yolo_utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class, non_max_suppression_rotated_bbox, get_batch_statistics_rotated_bbox
from trainer.trainer_adaptive_fusion import AugmentImage

class MLSFTrainerYolo:
    
    def __init__(self, mlsf:Union[MLSFYolo, MLSFYolov8, MLSFMobilenet], 
                dataset_kwargs:dict, optimizer_kwargs:dict,
                trainer_kwargs:dict, lr_scheduler_kwargs:dict):
        
        self.mlsf = mlsf
        
        self.output_dir = trainer_kwargs['output_dir']
        self.is_training = trainer_kwargs["is_training"]
        self.first_val_epoch = trainer_kwargs["first_val_epoch"]
        self.metric_eval_mode = trainer_kwargs["metric_eval_mode"]
        self.metric_average_mode = trainer_kwargs["metric_average_mode"]
        self.epochs = trainer_kwargs["epochs"]
        self.monitor_train = trainer_kwargs["monitor_train"]
        self.monitor_val = trainer_kwargs["monitor_val"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        
        self.checkpoint_idx = trainer_kwargs['checkpoint_idx']     
        self.robustness_augmentations = trainer_kwargs['robustness_augmentations']        
        self.gradient_accumulation_steps = trainer_kwargs['gradient_accumulation_steps']
        self.p_modality_dropout = trainer_kwargs['p_modality_dropout']
        self.modality_dropout = trainer_kwargs['modality_dropout']
        self.modality_corrupt = trainer_kwargs['modality_corrupt']
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)                
        
        self.logger = Logger(trainer_kwargs) 
        
        self._init_dataloader(dataset_kwargs)

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10          

        self.logger.log_line()
        self.logger.log_message(f'Train Dataloader:')
        self.logger.log_new_line()
        
        if type(self.train_dataloader.dataset) == Kitti2DObjectDetectDataset:
            self.logger.log_message(f'  Training on KiTTi Dataset   ')
            self.logger.log_message(f'LiDAR Dir: {self.train_dataloader.dataset.lidar_dir}')        
            self.logger.log_message(f'Calibration Dir: {self.train_dataloader.dataset.calibration_dir}')
            self.logger.log_message(f'Left Image Dir: {self.train_dataloader.dataset.left_image_dir}')
            self.logger.log_message(f'Right Image Dir: {self.train_dataloader.dataset.right_image_dir}')
            self.logger.log_message(f'Labels Dir: {self.train_dataloader.dataset.labels_dir}')
            self.logger.log_message(f'Train Batch Size: {self.train_dataloader.batch_size}')
            self.logger.log_message(f'Train Apply Augmentation: {self.train_dataloader.collate_fn.apply_augmentation}')
        elif type(self.train_dataloader.dataset) == NuScenesObjectDetectDataset:
            self.logger.log_message(f'  Training on NuScenes Dataset   ')
            self.logger.log_message(f'  Blobs Files: {self.train_dataloader.dataset.table_blob_paths}')
            self.logger.log_message(f'Train Batch Size: {self.train_dataloader.batch_size}')
        
        self.logger.log_line()
        
        self.logger.log_line()
        self.logger.log_message(f'Validation Dataloader:')
        self.logger.log_new_line()
        
        if type(self.validation_dataloader.dataset) == Kitti2DObjectDetectDataset:
            self.logger.log_message(f'LiDAR Dir: {self.validation_dataloader.dataset.lidar_dir}')        
            self.logger.log_message(f'Calibration Dir: {self.validation_dataloader.dataset.calibration_dir}')
            self.logger.log_message(f'Left Image Dir: {self.validation_dataloader.dataset.left_image_dir}')
            self.logger.log_message(f'Right Image Dir: {self.validation_dataloader.dataset.right_image_dir}')
            self.logger.log_message(f'Labels Dir: {self.validation_dataloader.dataset.labels_dir}')
            self.logger.log_message(f'Train Batch Size: {self.validation_dataloader.batch_size} - Ten Percent Train Log {self.ten_percent_train_batch}')
            self.logger.log_message(f'Validation Apply Augmentation: {self.validation_dataloader.collate_fn.apply_augmentation}')
        elif type(self.validation_dataloader.dataset) == NuScenesObjectDetectDataset:
            self.logger.log_message(f'  Training on NuScenes Dataset   ')
            self.logger.log_message(f'  Blobs Files: {self.validation_dataloader.dataset.table_blob_paths}')
            self.logger.log_message(f'Train Batch Size: {self.validation_dataloader.batch_size}')
            
        self.logger.log_line()

        self._init_optimizer(optimizer_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'  Optimizer: {self.optimizer.__class__.__name__}  ')        
        self.logger.log_new_line()
        
        if lr_scheduler_kwargs:
            self._init_lr_scheduler(lr_scheduler_kwargs)            
        
        self.logger.log_line()
        self.logger.log_message(f'MLSF Image Backbone: {self.mlsf.image_backbone.__class__.__name__} -- {self.mlsf.image_backbone_device} ')
        self.logger.log_message(f'MLSF LiDAR Backbone: {self.mlsf.lidar_backbone.__class__.__name__} -- {self.mlsf.lidar_backbone_device} ')
        
        if self.mlsf.apply_adaptive_fusion:
            self.logger.log_message(f'MLSF Fusion Backbone: {self.mlsf.adaptive_fusion_module.__class__.__name__} -- {self.mlsf.adaptive_fusion_device} ')
        else:
            self.logger.log_message(f'MLSF Fusion Backbone: {self.mlsf.apply_adaptive_fusion} ')
        
        if type(self.mlsf) == MLSFYolov8:            
            self.logger.log_message(f'  MLSF Detection head: {self.mlsf.detection_heads.__class__.__name__} -- {self.mlsf.adaptive_fusion_device}')
            self.logger.log_new_line()            
            self.logger.log_message(f'MLSF Image Size: {self.mlsf.image_width}x{self.mlsf.image_height}')

        self.logger.log_new_line()

    def _init_dataloader(self, dataset_kwargs:dict):

        if dataset_kwargs['_type'] == 'kitti':
            self._init_dataloader_kitti(dataset_kwargs)

        if dataset_kwargs['_type'] == 'nuscenes':
            self._init_dataloader_nuscenes(dataset_kwargs)            

    def _init_dataloader_nuscenes(self, dataset_kwargs:dict):

        def create_dataloader(kwargs:dict, image_resize:tuple, lidar_map_type:str, 
                            return_augment_loader:bool=False):

            dataset = NuScenesObjectDetectDataset(
                table_blob_paths=kwargs['table_blob_paths'], 
                root_dir=kwargs['root_dir']
            )

            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=kwargs['batch_size'], 
                collate_fn=NuScenesMLSFCollateFn(
                    image_resize=image_resize,                    
                    lidar_map_type=lidar_map_type   
                )             
            )

            return dataloader

        if dataset_kwargs['nuscenes_dataset_kwargs']['nuscenes_trainer_dataset_kwargs']:
            self.train_dataloader = create_dataloader(
                dataset_kwargs['nuscenes_dataset_kwargs']['nuscenes_trainer_dataset_kwargs'], 
                tuple(dataset_kwargs['image_resize']), 
                dataset_kwargs['lidar_map_type'],
                return_augment_loader=self.modality_corrupt
            )                    
            self.train_batch_size = self.train_dataloader.batch_size
            
        else:
            self.logger.log_line()
            self.logger.log_message(
                f'Trainer Kwargs not Found: {dataset_kwargs["trainer_kwargs"]}'
            )
            exit(1)
        
        if dataset_kwargs['nuscenes_dataset_kwargs']['nuscenes_validation_dataset_kwargs']:
            self.validation_dataloader = create_dataloader(
                dataset_kwargs['nuscenes_dataset_kwargs']['nuscenes_validation_dataset_kwargs'], 
                tuple(dataset_kwargs['image_resize']), 
                dataset_kwargs['lidar_map_type'],
            )
            self.val_batch_size = self.validation_dataloader.batch_size
        else:
            self.validation_dataloader = None
    
    def _init_dataloader_kitti(self, dataset_kwargs:dict):
        
        def create_dataloader(kwargs:dict, image_resize:tuple, lidar_map_type:str, 
                            return_augment_loader:bool=False):
            dataset = Kitti2DObjectDetectDataset(
                lidar_dir=kwargs['lidar_dir'],
                calibration_dir=kwargs['calibration_dir'],
                left_image_dir=kwargs['left_image_dir'],
                right_image_dir=kwargs['right_image_dir'],
                labels_dir=kwargs['labels_dir']
            )
            
            if type(self.mlsf) == MLSFYolo or type(self.mlsf) == MLSFYolov8:
                if return_augment_loader:
                    dataloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=kwargs['batch_size'], 
                        collate_fn=KitiiMLSFCollateAugment(
                            image_resize=image_resize,
                            detection_head='yolo'                        
                        ),
                        shuffle=True
                    )
                else:
                    dataloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=kwargs['batch_size'], 
                        collate_fn=KittiMLSFCollateFn(
                            image_resize=image_resize,
                            detection_head='yolo', 
                            lidar_map_type=lidar_map_type
                        ),
                        shuffle=True
                    )

                return dataloader
            
            elif type(self.mlsf) == MLSFMobilenet:
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=kwargs['batch_size'], 
                    collate_fn=KittiMLSFMobilenet(
                        image_resize=image_resize,
                        detection_head='yolo', 
                        lidar_map_type=lidar_map_type
                    ),
                    shuffle=True
                )

            return dataloader
        
        if dataset_kwargs['kitti_dataset_kwargs']['kitti_trainer_dataset_kwargs']:
            self.train_dataloader = create_dataloader(
                dataset_kwargs['kitti_dataset_kwargs']['kitti_trainer_dataset_kwargs'], 
                tuple(dataset_kwargs['image_resize']), 
                dataset_kwargs['lidar_map_type'],
                return_augment_loader=self.modality_corrupt
            )                    
            self.train_batch_size = self.train_dataloader.batch_size
            
        else:
            self.logger.log_line()
            self.logger.log_message(
                f'Trainer Kwargs not Found: {dataset_kwargs["trainer_kwargs"]}'
            )
            exit(1)
        
        if dataset_kwargs['kitti_dataset_kwargs']['kitti_validation_dataset_kwargs']:
            self.validation_dataloader = create_dataloader(
                dataset_kwargs['kitti_dataset_kwargs']['kitti_validation_dataset_kwargs'], 
                tuple(dataset_kwargs['image_resize']), 
                dataset_kwargs['lidar_map_type'],
            )
            self.val_batch_size = self.validation_dataloader.batch_size
        else:
            self.validation_dataloader = None            
        
    def _init_optimizer(self, optimizer_kwargs:dict):
        
        params_dict = []
        
        if optimizer_kwargs['tune_image_backbone']:        
            params_dict.append({
                'params':self.mlsf.image_backbone.parameters(), 'lr':optimizer_kwargs['image_backbone_lr'], 'model_name':f'ImageBackbone_{self.mlsf.image_backbone.__class__.__name__}'
            }) #detection head within image_backbone class for yolov3; not for yolov8
        
        if self.mlsf.use_lidar_backbone and optimizer_kwargs['tune_lidar_backbone']:
            params_dict.append({
                'params':self.mlsf.lidar_backbone.parameters(), 'lr':optimizer_kwargs['lidar_backbone_lr'], 'model_name':f'LiDARBackbone_{self.mlsf.lidar_backbone.__class__.__name__}'
            })
        
        if self.mlsf.apply_adaptive_fusion:
            params_dict.append({
                'params':self.mlsf.adaptive_fusion_module.parameters(), 'lr':optimizer_kwargs['adaptive_fusion_lr'], 'model_name':f'AdaptiveFusionModule'
            })
            
        if type(self.mlsf) == MLSFYolov8 or type(self.mlsf) == MLSFMobilenet:
            params_dict.append({
                'params':self.mlsf.detection_heads.parameters(), 'lr':optimizer_kwargs['ssd_head_lr'], 'model_name':f'Detection Head'
            })
            
            # params_dict.append({
            #     'params':self.mlsf.detection_heads.parameters(), 'lr':optimizer_kwargs['ssd_head_lr'], 'model_name':f' Detection Head'
            # })
        
        if optimizer_kwargs['_type'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                params_dict, 
                momentum=optimizer_kwargs['momentum']
            )
            
        if optimizer_kwargs['_type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                params_dict,
                weight_decay=optimizer_kwargs['momentum']
            )

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        if lr_scheduler_kwargs['_type'] == "linear":
            lr_scheduler_kwargs = lr_scheduler_kwargs['linear_lr_kwargs']            
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=lr_scheduler_kwargs['start_factor'],
                end_factor=lr_scheduler_kwargs['end_factor'],
                total_iters=self.epochs
            )

            self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
            self.logger.log_message(f'LR Scheduler Start Factor: {lr_scheduler_kwargs["start_factor"]}')
            self.logger.log_message(f'LR Scheduler End Factor: {lr_scheduler_kwargs["end_factor"]}')
            self.logger.log_new_line()            

        elif lr_scheduler_kwargs['_type'] == "cosine":
            lr_scheduler_kwargs = lr_scheduler_kwargs['cosine_annealing_lr_kwargs']            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=lr_scheduler_kwargs['eta_min']
            )

            self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
            self.logger.log_message(f'LR Scheduler TMax: {self.epochs}')
            self.logger.log_message(f'LR Scheduler ETA Min: {lr_scheduler_kwargs["eta_min"]}')            
            self.logger.log_new_line()
        
    def train(self):

        self.logger.log_line()
        self.logger.log_message(
            f'Training: Max Epoch - {self.epochs} -- Tasks: {self.mlsf.task_type}'
        )
        self.logger.log_new_line()

        self.total_training_time = 0.0
        self.cur_epoch = 0   
        # self.best_score = 0.0
        self.best_score = defaultdict(float)

        for epoch in range(1, self.epochs + 1):
            self.cur_epoch = epoch
            self.logger.log_line()

            if self.monitor_train:
                self.train_one_epoch()

                if (self.cur_epoch + 1) % self.checkpoint_idx == 0:
                    ckpt_dir = f'{self.output_dir}/ckpt_{self.cur_epoch}'
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)

                    torch.save(
                        self.mlsf.state_dict(), f'{ckpt_dir}/ckpt-model.pt'
                    )

            if self.monitor_val:
                self.valid_one_epoch()
                    
    def train_one_epoch(self):
        
        self.mlsf.train()
        
        total_loss = 0.0 
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0
        ten_percent_metric_per_grid = defaultdict(lambda:defaultdict(int))

        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        for batch_idx, data_items in enumerate(train_iter):
            
            step_begin_time = time.time()

            if self.modality_dropout and self.mlsf.use_lidar_backbone:
                loss, loss_components = self.train_one_step_modality_drop(data_items)
            elif self.modality_corrupt and self.mlsf.use_lidar_backbone:
                loss, loss_components = self.train_one_step_modality_corrupt(data_items)
            else:
                
                if not torch.is_tensor(data_items['targets']) or data_items['targets'].numel() == 0:
                    continue
                
                try:
                    loss, loss_components, _ = self.train_one_step(data_items)
                except Exception as e:
                    self.logger.log_message(f"Error during training step: {e} skipping batch")
                    
                    for k, v in data_items.items():
                        if torch.is_tensor(v):
                            print(f'{k} {v.shape}')
                    
                    continue

            step_end_time = time.time()

            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx == self.train_dataloader.__len__() - 1):                

                self.optimizer.step()
                self.lr_scheduler.step()

                self.optimizer.zero_grad()                                            
                current_lr = self.optimizer.param_groups[0]['lr']                

            total_loss += loss.item()
            ten_percent_batch_total_loss += loss.item()

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)

            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch    

                message = f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f} -- current_lr: {current_lr}'
                self.logger.log_message(message=message)
                self.logger.log_new_line()
                
                self.log_task_metrics(loss_components)

                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0.0
                ten_percent_metric_per_grid = defaultdict(lambda:defaultdict(int))
                
        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {total_loss/self.total_train_batch:.4f} -- current_lr: {current_lr}'
        )

        # writer.close()
            
    def train_one_step(self, data_items:dict):

        with torch.set_grad_enabled(True):
            loss, loss_components, outputs = self.mlsf(
                data_items['images'],
                data_items['lidar_2d'] if self.mlsf.use_lidar_backbone else None,
                data_items['targets'] if "targets" in data_items else None, 
                data_items['targets_3d'] if "targets_3d" in data_items else None,
                # cls_loss_type="focal" if self.cur_epoch > 20 else 'bce'
                cls_loss_type="focal"
            )

            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.mlsf.parameters(), self.gradient_clipping)                 

        return loss, loss_components, outputs

    def train_one_step_modality_corrupt(self, data_items:dict):

        total_loss = 0
        loss_components_accum = {}

        with torch.set_grad_enabled(True):
            losses = []
            
            loss, loss_components, _ = self.mlsf(
                data_items['images'],
                data_items['lidar_2d'] if self.mlsf.use_lidar_backbone else None,
                data_items['targets'] if "targets" in data_items else None, 
                data_items['targets_3d'] if "targets_3d" in data_items else None,
                cls_loss_type="focal" if self.cur_epoch > 20 else 'bce'
            )
            losses.append(loss)

            # if random.random() > 0.5:
            data_items['images'] = AugmentImage()(
                data_items['images'], 'pixelate'
            ) if random.random() > 0.5 else AugmentImage()(
                data_items['images'], 'SaltPapperNoise'
            )
            # else:
            #     data_items['lidar_2d'] = data_items['augmented_lidar_2d']
                
            loss, loss_components, _ = self.mlsf(
                data_items['images'],
                data_items['lidar_2d'] if self.mlsf.use_lidar_backbone else None,
                data_items['targets'] if "targets" in data_items else None, 
                data_items['targets_3d'] if "targets_3d" in data_items else None,
                cls_loss_type="focal" if self.cur_epoch > 20 else 'bce'
            )
            # Compute average loss across cases
            total_loss = sum(losses) / len(losses)

            # Unified backward pass
            with torch.autograd.set_detect_anomaly(True):
                total_loss.backward()               

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.mlsf.parameters(), self.gradient_clipping)
            
            total_loss = sum(losses) / len(losses)
            
        return total_loss, {k: v / len(losses) for k, v in loss_components_accum.items()}  # Normalize loss components
    
    def train_one_step_modality_drop(self, data_items: dict):

        total_loss = 0
        loss_components_accum = {}

        with torch.set_grad_enabled(True):
            losses = []

            # Image + LiDAR detection (Full Modality)
            loss, loss_components, _ = self.mlsf(
                data_items['images'],
                data_items['lidar_2d'] if self.mlsf.use_lidar_backbone else None,
                data_items['targets'] if "targets" in data_items else None, 
                data_items['targets_3d'] if "targets_3d" in data_items else None,
                cls_loss_type="focal" if self.cur_epoch > 20 else 'bce'
            )
            losses.append(loss)

            # Initialize loss components accumulation
            loss_components_accum = {k: v for k, v in loss_components.items()}

            # Modality-Dropped Training Step
            image_input = data_items['images'] if random.random() > self.p_modality_dropout else None
            lidar_input = data_items['lidar_2d'] if (random.random() > self.p_modality_dropout and self.mlsf.use_lidar_backbone) else None

            # Ensure at least one modality is present
            if image_input is None and lidar_input is None:
                image_input = data_items['images']  # Default to image-only

            # Compute loss with dropped modality
            loss, loss_components, _ = self.mlsf(image_input, lidar_input, data_items['targets'])
            losses.append(loss)

            # Accumulate loss components
            for k in loss_components:
                loss_components_accum[k] += loss_components[k]

            # Compute average loss across cases
            total_loss = sum(losses) / len(losses)

            # Unified backward pass
            with torch.autograd.set_detect_anomaly(True):
                total_loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.mlsf.parameters(), self.gradient_clipping)

        return total_loss, {k: v / len(losses) for k, v in loss_components_accum.items()}  # Normalize loss components
        
    def valid_one_epoch(self):
        
        def reshape_outputs(outputs:Iterable[torch.Tensor], img_size):
            
            for i, x in enumerate(outputs):

                bs, num_classes, grid_size_y, grid_size_x = x.shape
                stride = img_size // grid_size_y
                
                x = x.permute(0, 2, 3, 1).contiguous()
                
                grid = make_grid(grid_size_x, grid_size_y, x.device)                             
                
                x[..., 0:2] = (x[..., 0:2].sigmoid() + grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * stride # wh
                x[..., 4:] = x[..., 4:].sigmoid() # objectness_score, classes
                                    
                outputs[i] = x.view(bs, -1, num_classes) # number of outputs per anchor

            return torch.cat(outputs, 1)
        
        def make_grid(nx, ny, device):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid = torch.stack((xv, yv), 2).float().to(device)
            return grid        
        
        def apply_sigmoid_activation(outputs:list, img_size, anchor_grids):
                        
            for i,(x, anchor_grid) in enumerate(zip(outputs, anchor_grids)):         
                bs, num_anchors, grid_size_y, grid_size_x, num_classes = x.shape
                stride = img_size // x.size(2)

                grid = make_grid(grid_size_x, grid_size_y, x.device)                             
                
                x[..., 0:2] = (x[..., 0:2].sigmoid() + grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_grid # wh
                x[..., 4:] = x[..., 4:].sigmoid() # objectness_score, classes
                                    
                outputs[i] = x.view(bs, -1, num_classes) # number of outputs per anchor

            return torch.cat(outputs, 1)

        self.mlsf.eval()
        
        val_epoch_iter = tqdm(self.validation_dataloader, disable=True)   

        labels = defaultdict(list)
        # sample_metrics = {}  # List of tuples (TP, confs, pred)
        task_sample_metrics = defaultdict(list)

        if type(self.mlsf) == MLSFYolo:
            img_size = self.mlsf.image_backbone.hyperparams['height']
        
        elif type(self.mlsf) == MLSFMobilenet:
            img_size = self.mlsf.image_height
        
        elif type(self.mlsf) == MLSFYolov8:
            img_size = self.mlsf.image_height

        total_eval_loss = 0.0
        for batch_idx, data_items in enumerate(val_epoch_iter):
            
            if not torch.is_tensor(data_items['targets']) or data_items['targets'].numel() == 0:
                continue

            with torch.no_grad():
                try:
                    loss, loss_components, outputs = self.mlsf(
                        data_items['images'],
                        data_items['lidar_2d'] if self.mlsf.use_lidar_backbone else None,
                        data_items['targets'] if "targets" in data_items else None, 
                        data_items['targets_3d'] if "targets_3d" in data_items else None
                    )
                except Exception as e:
                    self.logger.log_message(f"Error during training step: {e} skipping batch")
                    for k, v in data_items.items():
                        if torch.is_tensor(v):
                            print(f'{k} {v.shape}')                    
                    continue
            
            total_eval_loss += loss.item()

            if type(self.mlsf) == MLSFYolo or type(self.mlsf) == MLSFMobilenet:

                for task_type in outputs:
                    if task_type == 'obj_2d':
                        
                        if not torch.is_tensor(data_items['targets']) or data_items['targets'].numel() == 0:
                            continue

                        targets = data_items['targets'].cpu()
                        labels[task_type] += targets[:, 1] #[class_id] 

                        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                        targets[:, 2:] *= img_size

                        if type(self.mlsf) == MLSFYolo:
                            anchor_grids = [yolo_layer.anchor_grid for yolo_layer in self.mlsf.image_backbone.yolo_layers]
                        elif type(self.mlsf) == MLSFMobilenet:
                            anchor_grids = [det_layer.anchor_grid for det_layer in self.mlsf.detection_heads.values()]
                        
                        outputs[task_type] = apply_sigmoid_activation(outputs[task_type], data_items['images'].size(2), anchor_grids)
                        outputs[task_type] = non_max_suppression(outputs[task_type])
                        task_sample_metrics[task_type] += get_batch_statistics(
                            outputs[task_type], targets, iou_threshold=0.5
                        )

                    elif 'obj_3d' in self.mlsf.task_type:
                        targets = data_items['targets_3d']
                        targets[:, 2:] *= img_size                        

                        outputs[task_type] = torch.cat(outputs[task_type], 1)
                        labels[task_type] += targets[:, 1] #[class_id] 

                        outputs[task_type] = non_max_suppression_rotated_bbox(outputs[task_type], conf_thres=0.5, nms_thres=0.5)
                        task_sample_metrics[task_type] += get_batch_statistics_rotated_bbox(outputs[task_type], 
                                                                        targets.to(outputs[task_type].device), 
                                                                        iou_threshold=0.5)

            elif type(self.mlsf) == MLSFYolov8:
                ious = outputs['ious']
                outputs = outputs['predictions']
                
                if batch_idx == 0:
                    for output in outputs:
                        print(output[0])                                    

                if self.mlsf.detection_type == 'anchor_free':
                    outputs = reshape_outputs(outputs, img_size)                
                else:
                    anchor_grids = [yolo_layer.anchor_grid for yolo_layer in self.mlsf.detection_heads.values()]             
                    outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)

        
        self.logger.log_new_line()
        self.logger.log_message(f'Epoch {self.cur_epoch} - Evaluation Loss {total_eval_loss/len(self.validation_dataloader):.4f}')    
        self.logger.log_line()        

        # Concatenate sample statistics
        
        for task_type, sample_metrics in task_sample_metrics.items():
            
            self.logger.log_message(f'  Computing Metrics for {task_type}   ')
            
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]            

            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_labels, labels[task_type]) 
            
            if type(self.validation_dataloader.dataset) == Kitti2DObjectDetectDataset:
                class_names = list(Enums.KiTTi_label2Id.keys())
            elif type(self.validation_dataloader.dataset) == NuScenesObjectDetectDataset:
                class_names = list(Enums.nuscenes_label2Id.keys())

            table_string = self.print_eval_stats(metrics_output, class_names, True)
            _, _, AP, _, _ = metrics_output        
            
            if AP.mean() > (self.best_score[task_type] + 0.02):
                self.best_score[task_type] = AP.mean()
                ckpt_dir = f'{self.output_dir}/best-model_{task_type}'
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                torch.save(
                    self.mlsf.state_dict(), f'{ckpt_dir}/best-model.pt'
                )
                with open(f'{ckpt_dir}/best-model.txt','w+') as f:
                    f.write(table_string)
                f.close()

                self.logger.log_message(f'Saving {task_type} Best Model at Performance - AP: {self.best_score}')
                self.logger.log_line()
            
    def print_eval_stats(self, metrics_output, class_names, verbose):
        
        table_string = ''
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
                        f'{AP[i]}:.5f',
                        f'{precision[i]:.5f}',
                        f'{recall[i]:.5f}',
                        f'{f1[i]:.5f}'
                    ])
                
                table_string = AsciiTable(ap_table).table
                
                self.logger.log_message(f'---------- mAP per Class----------')
                self.logger.log_message(f'{table_string}')
                self.logger.log_new_line()
                self.logger.log_message(f'---------- Total mAP {AP.mean():.5f} ----------')
                
        else:
            self.logger.log_message("---- mAP not measured (no detections found by model) ----")
            
        return table_string
    
    def log_task_metrics(self, loss_components:dict):
        
        for task_type in self.mlsf.task_type:

            self.logger.log_message(f'  Task: {task_type}   ')

            if task_type == 'obj_3d':
                for grid_size in loss_components[task_type]:
                    precision = loss_components[task_type][grid_size]['precision']
                    cls_acc = loss_components[task_type][grid_size]['cls_acc']
                    recall50 = loss_components[task_type][grid_size]['recall50']
                    recall75 = loss_components[task_type][grid_size]['recall75']
                    # iou_scores = ten_percent_metric_per_grid[grid_size]['iou_scores']
                    conf_obj = loss_components[task_type][grid_size]['conf_obj']
                    conf_noobj = loss_components[task_type][grid_size]['conf_noobj']
                
                    metrics_log = f'GridSize: {grid_size} -- Cls Acc: {cls_acc:.4f} Precision: {precision:.4f} Recall50: {recall50:.4f} Recall75: {recall75:.4f} Conf Obj: {conf_obj:.4f} Conf NoObj: {conf_noobj:.4f}'
                    self.logger.log_message(metrics_log)                                    
                
            elif task_type == 'obj_2d':
                lbox, lcls, lobj = loss_components[task_type]['lbox'], loss_components[task_type]['lcls'], loss_components[task_type]['lobj']
                self.logger.log_message(
                    f'  Loss BBox: {lbox:.4f} -- Loss Cls: {lcls:.4f} -- Loss Obj: {lobj:.4f}'
                )

            self.logger.log_new_line()