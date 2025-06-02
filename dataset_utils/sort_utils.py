
import numpy as np
import lap
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from collections import defaultdict

def linear_assignment(cost_matrix):
    try:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], y] for i in x if i>=0])
    except:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def compute_iou(bb_detections:np.array, bb_trackers:np.array):
    """
    bb_detections-detections of the frame. 
    bb_trackers-estimates of the frame.
    """
    
    bb_detections = np.expand_dims(bb_detections, 1)
    bb_trackers = np.expand_dims(bb_trackers, 0)
    
    xx1 = np.maximum(bb_detections[..., 0], bb_trackers[..., 0])
    yy1 = np.maximum(bb_detections[..., 1], bb_trackers[..., 1])
    xx2 = np.minimum(bb_detections[..., 2], bb_trackers[..., 2])
    yy2 = np.minimum(bb_detections[..., 3], bb_trackers[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    wh = w * h
    
    iou = wh / ((bb_detections[..., 2] - bb_detections[..., 0]) * (bb_detections[..., 3] - bb_detections[..., 1])                                      
    + (bb_trackers[..., 2] - bb_trackers[..., 0]) * (bb_trackers[..., 3] - bb_trackers[..., 1]) - wh)
    
    return iou

def convert_bbox_to_state(bbox:np.array):
    
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2
    y = bbox[1] + h/2
    
    s = w * h
    r = w/float(h)
    
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class Track:

    count = 0    #unique ID for each detected object. Incremented after creating Track object for a detection
    def __init__(self, bbox):
        '''
        bbox - (7,) shape (x, y, x', y', obj_score, class_idx, 0)
        
        dim_x=7: Specifies that the state vector (x) has 7 dimensions. 
            Based on common SORT implementations, this state is often: [cx, cy, s, r, dcx, dcy, ds] 
                where: cx, cy: center coordinates of the bounding box.
                    s: scale (area) of the bounding box.
                    r: aspect ratio of the bounding box (width/height).
        
            dcx, dcy, ds: respective velocities (changes per time step) of cx, cy, and s. The velocity of aspect ratio dr is often omitted or assumed constant, hence 7 dimensions.        
        
       dim_z=4: 
        Specifies that the measurement vector (z) has 4 dimensions. 
            This means the tracker directly measures/observes 4 variables from a detection, 
            which are typically [cx, cy, s, r] (center x, center y, scale, aspect ratio) derived from the detected bounding box.            
        '''
        
        self.id = Track.count
        self.kf = KalmanFilter(
            dim_x=7, 
            dim_z=4
        ) #dim_x = number of state variables, dim_z = number of measurement units. 
        
        self._init_kf()
        self.kf.x[:4] = convert_bbox_to_state(bbox)
        
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []
        self.det_class_idx = bbox[5]
        
        Track.count += 1
        
        self.centroidarr = []
        cx = (bbox[0]+bbox[2])//2
        cy = (bbox[1]+bbox[3])//2
        
        self.centroidarr.append((cx, cy))
        
        self.bbox_history = [bbox]
    
    def _init_kf(self):
        
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        # R: Covariance matrix of measurement noise (set to high for noisy inputs -> more 'inertia' of boxes')
        self.kf.R[2:,2:] *= 10. 
        #give high uncertainty to the unobservable initial velocities
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.        
        # Q: Covariance matrix of process noise (set to high for erratically moving things)
        self.kf.Q[-1,-1] *= 0.5 
        self.kf.Q[4:,4:] *= 0.5
        
    
    def update(self, bbox):
        """update state vector from estimate to observed bbox"""
        
        self.history = []
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # converts detection box (match) (x, y, x', y') to [cx, cy, s, r, dcx, dcy, ds]
        self.kf.update(convert_bbox_to_state(bbox))
        self.det_class_idx = bbox[5]
        cx = (bbox[0]+bbox[2])//2
        cy = (bbox[1]+bbox[3])//2
        
        self.centroidarr.append((cx, cy))
        self.bbox_history.append(bbox)        
    
    def predict(self):
        """Advances State vector using a linear constant velocity model and returns estimated bbox."""        
        
        # s_new = s_old + ds cannot be negative.

        if ((self.kf.x[2]+self.kf.x[6]) <= 0):
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if (self.time_since_update > 0):
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        # converts state [cx, cy, s, r, dcx, dcy, ds] to (x, y, x', y')
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """returns current bbox estimate"""

        arr_detclass = np.expand_dims(np.array([self.det_class_idx]), 0)

        arr_u_dot = np.expand_dims(self.kf.x[4], 0)
        arr_v_dot = np.expand_dims(self.kf.x[5], 0)
        arr_s_dot = np.expand_dims(self.kf.x[6], 0)

        return np.concatenate(
            (convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1
        )
    
class Sort:
    
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.4):
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0        
    
    def associate_detections_to_trackers(self, detections:np.array, trackers_arr:np.array):
        
        """
        1. Handle edge case, no trackers_arr then all detections are unmatched.
        2. Calculate IOU Matrix
            `iou_matrix[i, j]` will store the IOU between detection `i` and tracker `j`.
        
        3. Perform matching (Assignment)
            a. Each detection maps to at most one tracker. Each tracker maps to at most one detection
                (1-to-1) = (k, 2) (dim=1, det_idx, trk_idx)
            b. If not (a); each detection/tracker maps to more than one tracker/detection. 
                `linear_assignment` (typically the Hungarian algorithm) is used.
                It finds the assignment that maximizes the total IOU (or minimizes cost).
        
        4. Identify Unmatched Detections, a list of detections that don't match with any trackers. 
        5. Identify Unmatched Trackers, a list of trackers that don't match with any detections.
        
            if (4) and (5) not in matched indices.
        
        6. Filter out Matches with Low IOU (Crucial Post-Processing for Hungarian)
            The `linear_assignment` (Hungarian algorithm) will always try to find an assignment,
            even if the corresponding IOU is very low (or even zero if it's the "best" available option).
        
        7. Format Final `matches` Array (k, 2) (det_idx, trk_idx)
        8. Return matched, unmatched_detections and unmatched_trackers
        """
        
        if len(trackers_arr) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
                
        iou_matrix = compute_iou(detections, trackers_arr)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Since `linear_assignment` minimizes cost, and we want to maximize IOU,
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
                
        def unmatched(arr, unmatched_arr, axis):
            for i,_ in enumerate(arr):
                if (i not in matched_indices[:, axis]):
                    unmatched_arr.append(i)                    
            return unmatched_arr
            
        unmatched_detections = unmatched(detections, [], axis=0)
        unmatched_trackers = unmatched(trackers_arr, [], axis=1)            
        
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches):
            matches = np.concatenate(matches, axis=0)
        else:
            matches = np.empty((0, 2), dtype=int)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def update(self, dets=np.empty((0, 6))):
        """
        1. Get predicted locations from existing trackers 
        2. Delete any trackers if NaN.
        3. Associate Detections to Trackers --> matched, unmatched_dets, unmatched_trks
            associate_detections_to_trackers(dets, trks, self.iou_threshold)
            
        4. For matched detections --> update matched trackers with assigned detections. 
        5. Create and initialize new trackers for unmatched detections. 
        6. Remove dead tracks that have time_since_update > max_age. 
        """
        
        self.frame_count += 1
        
        trackers_arr = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        
        if len(self.trackers):
            for t, trk_arr in enumerate(trackers_arr):
                pos = self.trackers[t].predict()[0]
                trk_arr[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
                
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            
            trackers_arr = np.ma.compress_rows(np.ma.masked_invalid(trackers_arr))
            for t in reversed(to_del):
                self.trackers.pop(t)
                
        detections_for_sort = dets[:, :4]
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections_for_sort, 
                                                                                    trackers_arr)                
        
        if matched.shape[0] > 0:
            for m in matched:
                self.trackers[m[1]].update(dets[m[0], :])        
        
        for _det_idx in unmatched_dets:
            trk_bbox = np.hstack((dets[_det_idx], np.array([0])))
            tracker = Track(trk_bbox)
            self.trackers.append(tracker)
            
        i = len(self.trackers)
        for trk  in reversed(self.trackers):

            d = trk.get_state()[0]      
            # if (trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(
                    np.concatenate((d, [trk.id+1])).reshape(1, -1)
                )
            i -= 1

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret):
            return np.concatenate(ret)
            
        return np.empty((0, 9))
    
def trajectory_path(tracked_objects:np.array, active_track_ids:set, tracker_history:dict, frame_idx:int):
    
    if len(tracked_objects):
        for track in tracked_objects:
            x1, y1, x2, y2, cls_pred, conf, _, _, tracked_id = track
            active_track_ids.add(tracked_id)
            
            cx, cy = (x1+x2)//2, (y1+y2)//2
            
            tracker_history.setdefault(tracked_id, []).append(((cx, cy), frame_idx))
    
    trajectory = {}    
    for track_id, history in tracker_history.items():
        if track_id in active_track_ids:
            points = []
            
            for item in history:
                points.append(item[0])

            points = np.array(points, dtype=np.int32)            
            trajectory[track_id] = points
    
    inactive_ids = set(tracker_history.keys()) - active_track_ids
    for inactive_id in inactive_ids:
        del tracker_history[inactive_id]
    
    return trajectory

def count_objects(tracked_objects:np.array):
    
    object_counter = defaultdict(set)
    
    if len(tracked_objects):
        for track in tracked_objects:
            x1, y1, x2, y2, cls_pred, conf, _, _, tracked_id = track
            object_counter[cls_pred].add(tracked_id)
            
    return object_counter

def dwell_time(tracked_objects, track_lifetime:dict, frame_idx:int):
    
    if len(tracked_objects):
        for track in tracked_objects:
            
            x1, y1, x2, y2, cls_pred, conf, _, _, tracked_id = track
            if track_lifetime[tracked_id]['start'] is None:
                track_lifetime[tracked_id]['start'] = frame_idx
            
            track_lifetime[tracked_id]['end'] = frame_idx
            track_lifetime[tracked_id]['class'] = cls_pred
            
    return track_lifetime