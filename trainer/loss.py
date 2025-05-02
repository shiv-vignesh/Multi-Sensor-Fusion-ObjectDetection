import math
from typing import Iterable

import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon

def to_cpu(tensor):
    return tensor.detach().cpu()

# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Compute focal loss for binary classification.
    Inputs:
        inputs: raw logits, tensor of shape (N, *).
        targets: binary targets (0 or 1), same shape as inputs.
        alpha: balancing factor.
        gamma: focusing parameter.
        reduction: 'mean' or 'sum'.
    Returns:
        scalar loss.
    """
    # Compute binary cross-entropy loss with logits (no reduction)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # Get the probability of the true class
    pt = torch.exp(-bce_loss)
    loss = alpha * (1 - pt) ** gamma * bce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss    

def compute_loss(predictions, targets, model, eval_debug:bool=False, 
                model_type:str='yolov3', cls_loss_type:str="focal"):
    # Check which device was used
    device = targets.device

    # Add placeholder varables for the different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build yolo targets
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model, model_type)  # targets

    # Define different loss functions classification
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # Calculate losses for each yolo layer
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]
        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1)         
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            lbox += (1.0 - iou).mean()  # iou loss

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)                        
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1

                if cls_loss_type == "focal":
                    lcls += focal_loss(ps[:, 5:], t, alpha=0.25, gamma=2.0, reduction="mean")
                elif cls_loss_type == "bce":
                    # Use the tensor to calculate the BCE loss
                    lcls += BCEcls(ps[:, 5:], t)# BCE

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

    if model_type == 'yolov3':
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
    
    elif model_type == 'yolov8':
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
    
    if eval_debug:
        # print(f'lbox + lobj + lcls: {lbox} + {lobj} + {lcls}')
        pass

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))

def build_targets(p, targets, model, model_type:str='yolov3'):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = 3, targets.shape[0]  # number of anchors, targets #TODO
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    
    if model_type == 'yolov3':
        detection_layers = model.yolo_layers
    elif model_type == 'yolov8' or model_type == "mobilenet":
        #detection head module
        detection_layers = model.values()
    
    # for i, yolo_layer in enumerate(model.yolo_layers):
    for i, yolo_layer in enumerate(detection_layers):
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        anchors = yolo_layer.anchors / yolo_layer.stride
                
        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
        
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        t = targets * gain
        # Check if we have targets
        if nt:
            # Calculate ration between anchor and target box for both width and height
            r = t[:, :, 4:6] / anchors[:, None]
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            t = t[j]
        else:
            t = targets[0]

        # Extract image id in batch and class id
        b, c = t[:, :2].long().T
        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]  # grid wh
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long()
        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices

        # Convert anchor indexes to int
        a = t[:, 6].long()
        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # Add correct anchor for each target to the list
        anch.append(anchors[a])
        # Add class for each target to the list
        tcls.append(c)

    return tcls, tbox, indices, anch

def feature_alignment_loss(yolo_feat:torch.tensor, lidar_feat:torch.tensor):
    def compute_statistics(features):
        # Compute channel-wise mean and covariance
        mean = torch.mean(features, dim=[2, 3])  # (B, C)
        feat_flat = features.view(features.size(0), features.size(1), -1)  # (B, C, H*W)
        cov = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / feat_flat.size(2)  # (B, C, C)
        return mean, cov
    
    yolo_mean, yolo_cov = compute_statistics(yolo_feat)
    lidar_mean, lidar_cov = compute_statistics(lidar_feat)
    
    mean_loss = torch.nn.functional.mse_loss(yolo_mean, lidar_mean)
    cov_loss = torch.norm(yolo_cov - lidar_cov, p='fro') / yolo_cov.size(0)
    
    # 2. Local Structure Preservation
    def compute_local_structure(features):
        # Compute pairwise distances in feature space
        feat_flat = features.view(features.size(0), features.size(1), -1)  # (B, C, H*W)
        feat_norm = torch.norm(feat_flat, p=2, dim=1, keepdim=True)  # (B, 1, H*W)
        feat_normalized = feat_flat / (feat_norm + 1e-7)  # (B, C, H*W)
        similarity_matrix = torch.bmm(feat_normalized.transpose(1, 2), 
                                    feat_normalized)  # (B, H*W, H*W)
        return similarity_matrix
    
    yolo_structure = compute_local_structure(yolo_feat)
    point_structure = compute_local_structure(lidar_feat)
    
    structure_loss = torch.nn.functional.mse_loss(yolo_structure, point_structure)
    
    return mean_loss + 0.1 * cov_loss + 0.1 * structure_loss

def build_targets_3d(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.ByteTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)
    
    device = pred_boxes.device

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0).to(device)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1).to(device)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    th = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tim = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tre = FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0).to(device)

    # Convert to position relative to box
    target_boxes = target[:, 2:8]
    
    gxy = target_boxes[:, :2] * nG
    gwh = target_boxes[:, 2:4] * nG
    gimre = target_boxes[:, 4:]

    # Get anchors with best iou
    ious = torch.stack([rotated_box_wh_iou_polygon(anchor, gwh, gimre) for anchor in anchors])    

    best_ious, best_n = ious.max(0)
    b, target_labels = target[:, :2].long().t()
    
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gim, gre = gimre.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # Im and real part 
    tim[b, best_n, gj, gi] = gim
    tre[b, best_n, gj, gi] = gre

    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    rotated_iou_scores = rotated_box_11_iou_polygon(pred_boxes[b, best_n, gj, gi], target_boxes, nG)
    iou_scores[b, best_n, gj, gi] = rotated_iou_scores.to(device)
     
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf

def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / (box.union(b).area + 1e-12) for b in boxes]

    return np.array(iou, dtype=np.float32)

# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    return bev_corners
        
def rotated_bbox_iou_polygon(box1, box2):
    box1 = to_cpu(box1).numpy()
    box2 = to_cpu(box2).numpy()

    x,y,w,l,im,re = box1
    angle = np.arctan2(im, re)
    bbox1 = np.array(get_corners(x, y, w, l, angle)).reshape(-1,4,2)
    bbox1 = convert_format(bbox1)

    bbox2 = []
    for i in range(box2.shape[0]):
        x,y,w,l,im,re = box2[i,:]
        angle = np.arctan2(im, re)
        bev_corners = get_corners(x, y, w, l, angle)
        bbox2.append(bev_corners)
    bbox2 = convert_format(np.array(bbox2))

    return compute_iou(bbox1[0], bbox2)        
        
def rotated_box_wh_iou_polygon(anchor, wh, imre):
    w1, h1, im1, re1 = anchor[0], anchor[1], anchor[2], anchor[3]

    wh = wh.t()
    imre = imre.t()
    w2, h2, im2, re2 = wh[0], wh[1], imre[0], imre[1]

    anchor_box = torch.cuda.FloatTensor([100, 100, w1, h1, im1, re1]).view(-1, 6)    
    target_boxes = torch.cuda.FloatTensor(w2.shape[0], 6).fill_(100)

    target_boxes[:, 2] = w2
    target_boxes[:, 3] = h2
    target_boxes[:, 4] = im2
    target_boxes[:, 5] = re2

    ious = rotated_bbox_iou_polygon(anchor_box[0], target_boxes)

    return torch.from_numpy(ious)            

def rotated_box_11_iou_polygon(box1, box2, nG):

    box1_new = torch.cuda.FloatTensor(box1.shape[0], 6).fill_(0)
    box2_new = torch.cuda.FloatTensor(box2.shape[0], 6).fill_(0)

    box1_new[:, :4] = box1[:, :4]
    box1_new[:, 4:] = box1[:, 4:]

    box2_new[:, :4] = box2[:, :4] * nG
    box2_new[:, 4:] = box2[:, 4:]

    ious = []
    for i in range(box1_new.shape[0]):
        bbox1 = box1_new[i]
        bbox2 = box2_new[i].view(-1, 6)

        iou = rotated_bbox_iou_polygon(bbox1, bbox2).squeeze()
        ious.append(iou)

    ious = np.array(ious)

    return torch.from_numpy(ious)