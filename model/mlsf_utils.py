import torch
import numpy as np

import matplotlib.pyplot as plt

def parse_mobilenet_cfg(path:str):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
        lines = [x.strip() for x in lines if x and not x.startswith('#')]  # Remove comments and empty lines

    layers = []
    layer_dict = {}

    for line in lines:
        if line.startswith('['):  # New Layer
            if layer_dict:  # Save the previous layer
                layers.append(layer_dict)
                layer_dict = {}
            layer_dict['type'] = line[1:-1].strip()  # Extract layer type
        else:
            key, value = line.split('=')
            layer_dict[key.strip()] = value.strip()

    layers.append(layer_dict)  # Append last layer
    return layers

def generate_anchor_boxes(feature_map_size:tuple, image_size:tuple, scales:tuple, aspect_ratios:tuple):
    
    fm_h, fm_w = feature_map_size
    img_h, img_w = image_size
        
    anchor_boxes = []
    
    stride_h = img_h/fm_h
    stride_w = img_w/fm_w
    
    for i in range(fm_w):
        for j in range(fm_h):
            
            cx = (i + 0.5) * stride_w
            cy = (j + 0.5) * stride_h 
            
            for scale in scales:
                for ar in aspect_ratios:
                    w = scale * img_w * (ar ** 0.5)
                    h = scale * img_h / (ar ** 0.5)
                    
                    # h = scale * img_h
                    # w = h * ar
                    
                    anchor_boxes.append([cx, cy, w, h])
                    
    return torch.tensor(anchor_boxes)

def decode_boxes(anchor_boxes:torch.tensor, pred_loc:torch.tensor):
    
    batch_size = pred_loc.shape[0]
    dboxes = anchor_boxes.unsqueeze(0).expand(batch_size, -1, -1)
    
    pred_cx = dboxes[:, :, 0] + pred_loc[:, :, 0] * dboxes[:, :, 2]
    pred_cy = dboxes[:, :, 1] + pred_loc[:, :, 1] * dboxes[:, :, 3]
    pred_w = dboxes[:, :, 2] * torch.exp(pred_loc[:, :, 2])
    pred_h = dboxes[:, :, 3] * torch.exp(pred_loc[:, :, 3])

    xmin = pred_cx - 0.5 * pred_w
    ymin = pred_cy - 0.5 * pred_h
    xmax = pred_cx + 0.5 * pred_w
    ymax = pred_cy + 0.5 * pred_h
    
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)
    return boxes    

def box_center_to_corner(boxes, image_width=None, image_height=None):
    """
    Convert boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax).
    boxes: Tensor of shape (N, 4)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    x_min = cx - (w / 2)
    y_min = cy - (h / 2)
    x_max = cx + (w / 2)
    y_max = cy + (h / 2)

    # Clamp values within image bounds if specified
    if image_width is not None and image_height is not None:
        x_min = torch.clamp(x_min, min=0, max=image_width)
        y_min = torch.clamp(y_min, min=0, max=image_height)
        x_max = torch.clamp(x_max, min=0, max=image_width)
        y_max = torch.clamp(y_max, min=0, max=image_height)

    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

def compute_iou(boxes1, boxes2):
    """
    Compute the IoU between each pair of boxes in boxes1 and boxes2.
    boxes1: Tensor of shape (N, 4) in corner format.
    boxes2: Tensor of shape (M, 4) in corner format.
    Returns:
        iou: Tensor of shape (N, M)
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
    
    inter_xmin = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_ymin = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_xmax = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_ymax = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h
    
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area

def match_anchors(anchor_boxes:torch.tensor, gt_data:torch.tensor, 
                image_w:int, image_h:int, iou_threshold:float=0.5):
    
    num_defaults = anchor_boxes.size(0)
    num_gt = gt_data.size(0)
    
    target_offsets = torch.zeros(num_defaults, 4).to(anchor_boxes.device)
    target_labels = torch.zeros(num_defaults, dtype=torch.long).to(anchor_boxes.device)
    pos_mask = torch.zeros(num_defaults, dtype=torch.bool).to(anchor_boxes.device)
    
    anchor_boxes_center = anchor_boxes.clone()
    
    anchor_boxes_corner = box_center_to_corner(anchor_boxes_center, image_w, image_h)    
    gt_boxes = box_center_to_corner(gt_data[:, 1:], image_w, image_h)
    
    ious = compute_iou(anchor_boxes_corner, gt_boxes)  # shape: (num_defaults, num_gt)
    max_ious, max_idx = ious.max(dim=1)

    pos_mask = max_ious > iou_threshold
    target_labels[pos_mask] = gt_data[max_idx[pos_mask], 0].long()
    
    if pos_mask.sum() > 0:
        
        assigned_gt = gt_data[max_idx[pos_mask], 1:]  # (num_pos, 4)        
        default_pos = anchor_boxes_center[pos_mask]
        
        # This normalizes the difference between the ground truth center and the anchor’s center 
        # by the anchor’s width and height, respectively.
        # Taking the logarithm of the ratio helps stabilize training by compressing the scale differences.
        target_offsets[pos_mask, 0] = (assigned_gt[:, 0] - default_pos[:, 0]) / default_pos[:, 2]
        target_offsets[pos_mask, 1] = (assigned_gt[:, 1] - default_pos[:, 1]) / default_pos[:, 3]
        target_offsets[pos_mask, 2] = torch.log(assigned_gt[:, 2] / default_pos[:, 2] + 1e-8)
        target_offsets[pos_mask, 3] = torch.log(assigned_gt[:, 3] / default_pos[:, 3] + 1e-8)
    
    return target_offsets, target_labels, pos_mask

def batch_match_anchors(anchor_boxes:torch.tensor, targets:torch.tensor, 
                        batch_size:int, image_w:int, image_h:int, iou_threshold:float=0.5):
    
    num_defaults = anchor_boxes.size(0) #(total_anchors, 4)
    target_offsets_batch = []
    target_labels_batch = []
    pos_mask_batch = []
    
    for b in range(batch_size):
        gt_data = targets[targets[:, 0] == b][:, 1:] 
        
        if gt_data.numel() > 0:
            # denormalize
            gt_data[:, 1] *= image_w
            gt_data[:, 2] *= image_h
            gt_data[:, 3] *= image_w
            gt_data[:, 4] *= image_h            

            target_offsets, target_labels, pos_mask = match_anchors(
                anchor_boxes, gt_data, image_w, image_h, iou_threshold=iou_threshold
            )            

        else:
            target_offsets = torch.zeros(num_defaults, 4, device=anchor_boxes.device)
            target_labels = torch.zeros(num_defaults, dtype=torch.long, device=anchor_boxes.device)
            pos_mask = torch.zeros(num_defaults, dtype=torch.bool, device=anchor_boxes.device)
            
        target_offsets_batch.append(target_offsets)
        target_labels_batch.append(target_labels)
        pos_mask_batch.append(pos_mask)
        
    target_offsets_batch = torch.stack(target_offsets_batch, dim=0)
    target_labels_batch = torch.stack(target_labels_batch, dim=0)
    pos_mask_batch = torch.stack(pos_mask_batch, dim=0)
    
    return target_offsets_batch, target_labels_batch, pos_mask_batch

def compute_confidence_loss(pred_cls:torch.tensor, target_labels:torch.tensor, pos_mask:torch.tensor, neg_factor:int=3):
    
    bs, num_anchors, num_cls = pred_cls.shape    
    target_one_hot = torch.zeros(bs, num_anchors, num_cls, device=target_labels.device) 
    target_one_hot.scatter_(2, target_labels.unsqueeze(-1), 1)
    
    loss = (-target_one_hot * torch.nn.functional.log_softmax(pred_cls, dim=-1)).sum(dim=-1) #(bs, num_anchors)
    
    pos_loss = loss[pos_mask]
    neg_mask = torch.logical_not(pos_mask)
    neg_loss = loss[neg_mask]
    
    #hard-negative mining (topk hard negatives)
    num_pos = pos_mask.sum().clamp(min=1)
    neg_num = min(neg_loss.shape[0], neg_factor * num_pos.item())  # Select top-k negatives
    
    if neg_num > 0:
        neg_loss, _ = torch.topk(neg_loss, neg_num)

    # Final loss normalization
    cls_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

    return cls_loss

def ssd_loss(pred_loc:torch.tensor, pred_cls:torch.tensor, 
            target_offsets:torch.tensor, target_labels:torch.tensor,
            pos_mask:torch.tensor, alpha=0.5):
            
    assert pred_loc.shape[1] == target_offsets.shape[1], "Mismatch in target and predicted boxes"
    
    batch_size = pred_loc.shape[0]
    
    pred_loc_pos = pred_loc[pos_mask]
    target_offsets_pos = target_offsets[pos_mask]
        
    loc_loss = torch.nn.functional.smooth_l1_loss(
        pred_loc_pos, target_offsets_pos, reduction="sum"
    )
    
    num_pos = pos_mask.float().sum().clamp(min=1.0)    
    loc_loss /= num_pos
    
    cls_loss = compute_confidence_loss(pred_cls, target_labels, pos_mask)    
    print(f'Cls Loss: {cls_loss.item():.4f} Loc Loss: {loc_loss.item():.4f} PositiveSum: {num_pos.item()}')       
    
    total_loss = (cls_loss + alpha * loc_loss)
    return total_loss, cls_loss, loc_loss, alpha
     
    # pred_cls_flat = pred_cls.view(-1, pred_cls.size(-1))
    # target_labels_flat = target_labels.view(-1)        
    
    # cls_loss = torch.nn.functional.cross_entropy(
    #     pred_cls_flat, target_labels_flat, reduction="sum"
    # )
    
    
def compute_iou_single(box1, box2):
    """
    Compute IoU between two boxes in (xmin, ymin, xmax, ymax) format.
    """
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    inter_w = max(0, xmax - xmin)
    inter_h = max(0, ymax - ymin)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def compute_precision_recall_f1(det, gt, iou_threshold=0.5):
    """
    Compute precision, recall, and F1 score for detections of one class.
    
    Args:
        det: List of detection dicts for one class (each with 'box', 'score', 'image_id').
             Sorted by descending score.
        gt: List of ground truth boxes for one class (each with 'box', 'image_id').
        iou_threshold: IoU threshold to count a detection as true positive.
    
    Returns:
        precision, recall, f1, ap (average precision) for this class.
    """
    # Organize ground truths by image.
    gt_by_image = {}
    for g in gt:
        img = g['batch_id']
        if img not in gt_by_image:
            gt_by_image[img] = []
        gt_by_image[img].append(g['box'])
    
    # Mark ground truths as not detected.
    detected = {img: [False] * len(boxes) for img, boxes in gt_by_image.items()}
    
    tp = np.zeros(len(det))
    fp = np.zeros(len(det))
    
    for i, d in enumerate(det):
        img = d['batch_id']
        box_pred = d['box']
        if img in gt_by_image:
            gt_boxes = gt_by_image[img]
            ious = np.array([compute_iou_single(box_pred, gt_box) for gt_box in gt_boxes])
            if len(ious) > 0:
                max_iou = ious.max()
                max_idx = ious.argmax()
                if max_iou >= iou_threshold:
                    if not detected[img][max_idx]:
                        tp[i] = 1  # True positive
                        detected[img][max_idx] = True
                    else:
                        fp[i] = 1  # Duplicate detection
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1  # No ground truth in this image for this class.
    
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # Number of ground truths for this class.
    total_gt = sum([len(v) for v in gt_by_image.values()])
    if total_gt == 0:
        return 0, 0, 0, 0
    
    # If there are no detections, return zeros immediately.
    if len(det) == 0:
        return 0, 0, 0, 0    
    
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    
    # F1 score at each detection.
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Average precision: using the 11-point interpolation method (simplified)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        if p.size > 0:
            p_val = p.max()
        else:
            p_val = 0
        ap += p_val / 11.0
    
    # Final precision, recall, and F1: use the last values.
    final_precision = precision[-1]
    final_recall = recall[-1]
    final_f1 = f1_scores[-1]
    
    return final_precision, final_recall, final_f1, ap

class MLSFDiagonistic:    
    
    def __init__(self):        
        
        self.pos_match_count = []
    
    def add_pos_mask(self, pos_mask:torch.tensor):
        
        pos_count = pos_mask.float().sum().item()
        self.pos_match_count.append(pos_count)
        
    def plot_pos_match_hist(self, save_path:str):
        
        plt.figure(figsize=(8, 6))
        plt.hist(self.pos_match_count, bins=30, edgecolor='black')
        plt.title("Distribution of Positive Anchors per Batch")
        plt.xlabel("Number of Positive Anchors")
        plt.ylabel("Frequency")
        
        plt.savefig(save_path)
        
    def plot_pos_match_line(self, save_path:str):
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.pos_match_count, marker='o')
        plt.title("Positive Anchors per Batch Over Training")
        plt.xlabel("Batch Index")
        plt.ylabel("Number of Positive Anchors")
        plt.grid(True)
        
        plt.savefig(save_path)
        
    def plot_gradient_flow(self, ave_grads:list, max_grads:list, layers:list, save_path:str):
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c", label="Max gradient")
        plt.bar(range(len(ave_grads)), ave_grads, alpha=0.1, lw=1, color="b", label="Avg gradient")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=0, top=max(max_grads)*1.1)
        plt.xlabel("Layers")
        plt.ylabel("Gradient value")
        plt.title("Gradient Flow")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
