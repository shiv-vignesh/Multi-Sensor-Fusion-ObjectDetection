import torch
import numpy as np
import torch.nn as nn
import torchvision

import time, math
from tqdm import tqdm
# import wandb

from trainer.loss import rotated_bbox_iou_polygon

# wandb.init(project="MLSF-Yolov8-Training") 

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression_rotated_bbox(prediction, conf_thres=0.25, nms_thres=0.0, topk=100):
    """
    Efficient NMS for rotated bboxes (only filtering + topk).
    
    Returns: (bs, -1, 8) => (x, y, w, l, im, re, object_conf, class_idx)
    """
    batch_size = len(prediction)
    device = prediction[0].device
    processed = []

    for image_pred in prediction:
        if image_pred is None or image_pred.size(0) == 0:
            processed.append(torch.zeros((0, 8), device=device))
            continue

        # Filter by object confidence threshold
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]
        if not image_pred.size(0):
            processed.append(torch.zeros((0, 8), device=device))
            continue

        # Compute score = obj_conf * max class score
        class_scores, class_preds = image_pred[:, 7:].max(1)
        score = image_pred[:, 6] * class_scores

        # Sort by score (descending)
        image_pred = image_pred[(-score).argsort()]
        class_preds = class_preds[(-score).argsort()]

        # Keep topk
        top_n = min(topk, image_pred.size(0))
        image_pred = image_pred[:top_n]
        class_preds = class_preds[:top_n].unsqueeze(1)

        # Final output format: (x, y, w, l, im, re, object_conf, class_idx)
        output = torch.cat([image_pred[:, :7], class_preds.float()], dim=1)
        processed.append(output)

    # Pad to match batch shape
    max_len = max(p.size(0) for p in processed)
    padded = []
    for p in processed:
        pad_len = max_len - p.size(0)
        if pad_len > 0:
            pad = torch.zeros((pad_len, 8), device=device)
            p = torch.cat([p, pad], dim=0)
        padded.append(p.unsqueeze(0))

    return torch.cat(padded, dim=0)  # shape: (bs, max_len, 8)


def non_max_suppression_rotated_bbox_2(prediction, conf_thres=0.95, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]
        if not image_pred.size(0):
            continue
        score = image_pred[:, 6] * image_pred[:, 7:].max(1)[0]
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 7:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), 1)
        
        print(detections.shape)
        exit(1)

        keep_boxes = []
        while detections.size(0):
            current_det = detections[0, :6]
            current_label = detections[0, -1]
            other_boxes = detections[:, :6]
            other_labels = detections[:, -1]

            # Compute overlap mask
            ious = fast_rotated_iou(current_det, other_boxes)
            large_overlap = torch.from_numpy((ious > nms_thres).astype('bool')).to(prediction.device)
            label_match = current_label == other_labels

            invalid = large_overlap & label_match
            weights = detections[invalid, 6:7]
            detections[0, :6] = (weights * detections[invalid, :6]).sum(0) / weights.sum()
            keep_boxes.append(detections[0])
            detections = detections[~invalid]

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def fast_rotated_iou(box1, boxes):
    """Compute IoU between one rotated box and many others using vectorized polygon math."""
    import cv2

    box1 = box1.cpu().numpy()
    boxes = boxes.cpu().numpy()

    x1, y1, w1, l1, im1, re1 = box1
    angle1 = np.arctan2(im1, re1) * 180 / np.pi
    rect1 = ((x1, y1), (w1, l1), angle1)
    poly1 = cv2.boxPoints(rect1).astype(np.float32)

    ious = []
    for i in range(boxes.shape[0]):
        x, y, w, l, im, re = boxes[i]
        angle = np.arctan2(im, re) * 180 / np.pi
        rect = ((x, y), (w, l), angle)
        poly2 = cv2.boxPoints(rect).astype(np.float32)

        # Use OpenCV for polygon intersection
        int_pts = cv2.intersectConvexConvex(poly1, poly2)[1]
        if int_pts is not None:
            inter_area = cv2.contourArea(int_pts)
            area1 = w1 * l1
            area2 = w * l
            union = area1 + area2 - inter_area
            iou = inter_area / union
        else:
            iou = 0.0
        ious.append(iou)

    return np.array(ious)

def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold, return_matching_preds:bool=False):
    """ Compute true positives, predicted scores and predicted labels per sample """
    """ return_matching_preds: temporary fix for 3D box visualization; current has high FP. """
    batch_metrics = []
    batch_detected_boxes = []
    
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :6]
        pred_scores = output[:, 6].detach().cpu()
        pred_labels = output[:, -1].detach().cpu()

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            # matching_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                ious = rotated_bbox_iou_polygon(pred_box, target_boxes)
                iou, box_index = torch.from_numpy(ious).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1                    
                    detected_boxes += [box_index]
                    # matching_boxes.append(pred_box)
                # print(f'True Positive Detected {sample_i}: {len(detected_boxes)} {len(target_boxes)}')

            batch_detected_boxes.append(detected_boxes)
        # print(f'True Positive Detected {sample_i}: {len(detected_boxes)} {len(target_boxes)} {len(pred_boxes)}')

        batch_metrics.append([true_positives, pred_scores, pred_labels])
    
    if return_matching_preds:
        return batch_metrics, batch_detected_boxes
    
    return batch_metrics

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue        
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf        

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded

    return output

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def get_batch_statistics_eval(outputs, targets, iou_threshold, difficulty_levels=None):

    """
    Compute true positives (TP), false positives (FP), false negatives (FN),
    predicted scores and predicted labels per sample.
    """
    batch_metrics = []
    stats_per_difficulty = []

    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        num_preds = pred_boxes.shape[0]
        true_positives = np.zeros(num_preds)
        false_positives = np.zeros(num_preds)

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        num_targets = len(annotations)
        detected_boxes = []
        false_negatives = []  # To store GT class labels for missed boxes
        
        # Difficulty tracking
        # if difficulty_levels is not None:
        #     sample_difficulties = difficulty_levels.get(sample_i, [])
        #     tp_per_level = {'easy': 0, 'moderate': 0, 'hard': 0}

        if num_targets:
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # Stop if all GTs already matched
                if len(detected_boxes) == num_targets:
                    break

                if pred_label not in target_labels:
                    false_positives[pred_i] = 1
                    continue

                filtered = [(i, t) for i, t in enumerate(target_boxes) if target_labels[i] == pred_label and i not in detected_boxes]
                if not filtered:
                    false_positives[pred_i] = 1
                    continue

                filtered_indices, filtered_targets = zip(*filtered)
                iou, best_idx = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)
                best_target_idx = filtered_indices[best_idx]

                if iou >= iou_threshold:
                    true_positives[pred_i] = 1
                    detected_boxes.append(best_target_idx)
                    # level = sample_difficulties[best_target_idx] if best_target_idx < len(sample_difficulties) else 'unknown'
                    # if level in tp_per_level:
                    #     tp_per_level[level] += 1
                                            
                else:
                    false_positives[pred_i] = 1                    
        else:
            false_positives = np.ones(num_preds)

        # Count false negatives (GT boxes not matched)
        false_negatives.extend(target_labels[i] for i in range(num_targets) if i not in detected_boxes)

        batch_metrics.append([true_positives, false_positives, false_negatives, pred_scores, pred_labels])
        # stats_per_difficulty.append(tp_per_level)
    return batch_metrics
    # if difficulty_levels is not None:
    #     return batch_metrics, stats_per_difficulty
    # else:
    #     return batch_metrics

def compute_stats_per_difficulty(outputs, targets, obj_levels, iou_threshold=0.45):
    """
    Compute true positives per difficulty level per sample.

    Parameters:
    - outputs: list of length B, each [N_i, 6] -> (x1, y1, x2, y2, score, cls)
    - targets: Tensor [M, 6] -> (batch_idx, cls, x1, y1, x2, y2)
    - obj_levels: List[str] of length M -> difficulty level for each target box
    - iou_threshold: IoU threshold for TP

    Returns:
    - stats_per_sample: List[Dict[difficulty, TP count]]
    """
    stats_per_sample = []

    for sample_i, output in enumerate(outputs):
        if output is None or len(output) == 0:
            stats_per_sample.append({})
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, 5].int()

        # Filter GTs for this sample
        sample_mask = targets[:, 0] == sample_i
        sample_targets = targets[sample_mask]
        sample_difficulties = [obj_levels[i] for i in range(len(obj_levels)) if sample_mask[i]]

        if len(sample_targets) == 0:
            stats_per_sample.append({})
            continue

        gt_labels = sample_targets[:, 1].int()
        gt_boxes = sample_targets[:, 2:6]

        detected_gt = set()
        tp_per_difficulty = {}

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            # Match candidate GTs
            matches = [(i, box) for i, (box, label) in enumerate(zip(gt_boxes, gt_labels))
                       if label == pred_label and i not in detected_gt]

            if not matches:
                continue

            match_indices, match_boxes = zip(*matches)
            ious = bbox_iou(pred_box.unsqueeze(0), torch.stack(match_boxes)).squeeze(0)
            best_iou, best_idx = ious.max(0)
            best_iou = best_iou.item()
            best_match_idx = match_indices[best_idx.item()]

            if best_iou >= iou_threshold:
                level = sample_difficulties[best_match_idx]
                tp_per_difficulty[level] = tp_per_difficulty.get(level, 0) + 1
                detected_gt.add(best_match_idx)

        stats_per_sample.append(tp_per_difficulty)

    return stats_per_sample


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-9)
    
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        
        union = b1_area + b2_area - inter_area + 1e-9
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-9
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-9
        
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + 1e-9  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + 1e-9) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-9  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

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

def reshape_outputs(outputs:list):
    
    import math
    
    num_anchors = 3
    
    for i, x in enumerate(outputs):
        bs, num_preds, _ = x.shape
        grid_size = int(math.sqrt(num_preds // num_anchors))
        outputs[i] = x.view(bs, num_anchors, grid_size, grid_size, -1)
        
    return outputs

def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def compute_detection_loss_v8(predictions: torch.Tensor, targets: torch.Tensor, 
                              num_classes: int, device: torch.device, 
                              cls_loss_type: str = "focal", alpha=0.25, gamma=2.0):
    """
    FIXME, alignment_metric = class_probs.sigmoid().max(1)[0] * iou 
    """
    # Initialize losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build YOLOv8 targets
    tcls, tbox, indices = build_targets_v8(predictions, targets)

    # Define Loss Functions
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    
    ious = []
    
    for layer_idx, layer_predictions in enumerate(predictions):
        B, C, H, W = layer_predictions.shape
        layer_predictions = layer_predictions.permute(0, 2, 3, 1).contiguous()
        
        # Get target grid cell indices
        b, grid_y, grid_x = indices[layer_idx]  # Select only relevant grid cells
        
        # Build empty objectness target tensor
        tobj = torch.zeros((B, H, W), device=device)  # Shape: (B, H, W)

        if b.shape[0] > 0:  # If there are matching targets
            # Select predictions for responsible grid cells
            ps = layer_predictions[b, grid_y, grid_x]  # Shape: (num_targets, C)

            box_xy = ps[:, :2].sigmoid()  # (x, y)
            box_wh = ps[:, 2:4].exp()  # (w, h)

            # Compute IoU Loss for Box Regression
            pbox = torch.cat((box_xy, box_wh), dim=1)  # (num_targets, 4)
            iou = bbox_iou(pbox, tbox[layer_idx], x1y1x2y2=False, CIoU=True)
            ious.append(iou.mean().item())

            lbox += (1.0 - iou).mean()
            # print("IoU stats: min {:.3f}, max {:.3f}, mean {:.3f}".format(
            #         iou.min().item(), iou.max().item(), iou.mean().item()))

            # Assign objectness target based on IoU
            tobj[b, grid_y, grid_x] = iou.detach().clamp(0).type(tobj.dtype)

            # Compute Classification Loss
            if num_classes > 1:
                t = torch.zeros_like(ps[:, 5:]  , device=device)  # One-hot encoded target
                t[range(b.shape[0]), tcls[layer_idx]] = 1  # Assign correct class index

                if cls_loss_type == "focal":
                    # BCE Loss (Manually computed since class_probs already has sigmoid applied)
                    # bce_loss = - (t * torch.log(class_probs + 1e-8) + (1 - t) * torch.log(1 - class_probs + 1e-8))
                    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(ps[:, 5:], t, reduction="none")
                    pt = torch.exp(-bce_loss)
                    bce_loss = alpha * (1 - pt) ** gamma * bce_loss
                    lcls += bce_loss.mean()
                else:
                    lcls += BCEcls(ps[:, 5:], t)

            # wandb.log({
            #     # 'class_scores':ps[:, 5:].sigmoid()
            #     f'class_{idx}':ps[:, 5+idx].sigmoid().mean().detach().cpu().item() for idx in range(num_classes)
            # })

        # Compute Objectness Loss (Only once)        
        lobj += BCEobj(layer_predictions[..., 4], tobj)
        # raw_obj_logits = layer_predictions[..., -1].sigmoid()
        # print("Raw objectness logits: mean {:.3f}, min {:.3f}, max {:.3f}".format(
        #     raw_obj_logits.mean().item(), raw_obj_logits.min().item(), raw_obj_logits.max().item()))                        

    # Scale Losses
    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Compute Total Loss
    loss = lbox + lobj + lcls
    
    # wandb.log(
    #     {"lbox": lbox.detach().cpu().item(), "lobj": lobj.detach().cpu().item(), "lcls": lcls.detach().cpu().item()}
    # )
    
    # wandb.log({
    #     'ious':sum(ious)/len(ious)        
    # })
    
    return loss, torch.cat((lbox.detach().cpu(), lobj.detach().cpu(), lcls.detach().cpu(), loss.detach().cpu())), ious

def build_targets_v8(predictions:torch.Tensor, target:torch.Tensor):
    ''' 
    Assign ground truth to grid cells of yolov8 (80x80, 40x40, 20x20)
    
    Args:
        predictions - List[tensor] --> feature maps of (bs, num_cls + 5, h, w)
        target - Tensor of shape (M, 6) --> (batch_idx, class_id, x, y, w, h)

    Returns:
        tcls (list): Class indices for selected grid cells. (target_class indices)
        tbox (list): Box coordinates for selected grid cells. (target bbox for selected grid cells)
        indices (list): (batch_id, grid_y, grid_x) for each grid scale. 
    '''
    
    tcls, tbox, indices = [], [], []    
    for layer_idx, layer_predictions in enumerate(predictions):
        _, _, H, W = layer_predictions.shape
        
        #helper tensor to compute grid dimension relevant targets (normalized target to grid_w and grid_h)
        gain = torch.tensor([1, 1, W, H, W, H], device=target.device).float()
        t_grid = target * gain #convert to grid space from normalized score 
        
        #get integer indices of grid cells. 
        gxy = t_grid[:, 2:4] #across all batches, get the xy values scaled to grid. 
        gij = gxy.long() #convert to long for indices 
        gi, gj = gij.T 
        
        # Store grid cell indices, box targets, and class targets
        indices.append((t_grid[:, 0].long(), gi.clamp_(0, W - 1), gj.clamp_(0, H - 1)))
        tbox.append(torch.cat((gxy - gij, t_grid[:, 4:6]), 1))  # Box offset
        tcls.append(t_grid[:, 1].long())  # Class ID     
        
    return tcls, tbox, indices