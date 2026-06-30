# core/bytetracker.py
import numpy as np

def ious(atlbrs, btlbrs):
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)))
    ious = np.zeros((len(atlbrs), len(btlbrs)))
    for i, box1 in enumerate(atlbrs):
        for j, box2 in enumerate(btlbrs):
            x1, y1, x2, y2 = box1
            x1_p, y1_p, x2_p, y2_p = box2
            inter_x1 = max(x1, x1_p)
            inter_y1 = max(y1, y1_p)
            inter_x2 = min(x2, x2_p)
            inter_y2 = min(y2, y2_p)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_p - x1_p) * (y2_p - y1_p)
            union_area = area1 + area2 - inter_area
            ious[i, j] = inter_area / (union_area + 1e-6)
    return ious

def iou_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    return 1 - _ious

class KalmanFilter:
    def __init__(self):
        self._dim_x = 7
        self._dim_z = 4
        self._F = np.eye(7, dtype=np.float32)
        self._F[0, 4] = 1.0
        self._F[1, 5] = 1.0
        self._F[2, 6] = 1.0
        
        self._H = np.zeros((4, 7), dtype=np.float32)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0
        
        self._P = np.eye(7, dtype=np.float32) * 10.
        self._P[4:, 4:] *= 1000.
        
        self._Q = np.eye(7, dtype=np.float32)
        self._Q[4:, 4:] *= 0.01
        self._Q[2, 2] *= 0.01
        
        self._R = np.eye(4, dtype=np.float32)
        self._R[2:, 2:] *= 10.

    def initiate(self, measurement):
        mean_pos = measurement.copy()
        mean_vel = np.zeros(3, dtype=np.float32)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)
        
        std = np.array([
            2.0, 2.0, 1e-2, 2.0,
            10.0, 10.0, 1e-5
        ], dtype=np.float32)
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean, covariance):
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + self._Q
        return mean, covariance

    def update(self, mean, covariance, measurement):
        I = np.eye(self._dim_x, dtype=np.float32)
        S = self._H @ covariance @ self._H.T + self._R
        K = covariance @ self._H.T @ np.linalg.inv(S)
        
        y = measurement - self._H @ mean
        mean = mean + K @ y
        covariance = (I - K @ self._H) @ covariance
        return mean, covariance

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    _count = 0
    
    def __init__(self, tlwh, score, class_id=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.class_id = class_id
        self.track_id = 0
        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False  # ← 关键：默认 False
        self.tracklet_len = 0

    def predict(self):
        if self.mean is not None and self.covariance is not None:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.is_activated = True  # ← 关键：激活时设为 True

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.state = TrackState.Tracked
        self.is_activated = True  # ← 确保更新时也激活
        self._tlwh = new_tlwh

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]
        return ret

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def end_frame(self):
        return self.frame_id

def joint_tracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if tid not in exists:
            exists[tid] = 1
            res.append(t)
    return res

def sub_tracks(tlista, tlistb):
    tracks = {}
    for t in tlista:
        tracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tid in tracks:
            del tracks[tid]
    return list(tracks.values())

def remove_duplicate_tracks(tracksa, tracksb):
    p_dist = iou_distance(tracksa, tracksb)
    pairs = np.where(p_dist < 0.15)
    dupa, dupb = [], []
    for p in zip(*pairs):
        if tracksa[p[0]].frame_id > tracksb[p[1]].frame_id:
            dupb.append(p[1])
        else:
            dupa.append(p[0])
    resa = [t for i, t in enumerate(tracksa) if i not in dupa]
    resb = [t for i, t in enumerate(tracksb) if i not in dupb]
    return resa, resb

class BYTETracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, dets, img_size, img_size_ori):
        self.frame_id += 1
        activated_tracks = []
        lost_tracks = []
        removed_tracks = []

        if len(dets) == 0:
            for track in self.tracked_tracks:
                track.mark_lost()
                lost_tracks.append(track)
            self.tracked_tracks = []
            self.lost_tracks = joint_tracks(self.lost_tracks, lost_tracks)
            self._prune_lost_tracks(removed_tracks)
            return []

        scores = dets[:, 4]
        bboxes = dets[:, :4]
        class_ids = dets[:, 5].astype(int) if dets.shape[1] > 5 else np.full(len(dets), -1, dtype=int)
        remain_inds = scores > self.track_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        class_ids_keep = class_ids[remain_inds]

        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), score, int(cls_id))
                for tlbr, score, cls_id in zip(dets, scores_keep, class_ids_keep)
            ]
        else:
            detections = []

        track_pool = joint_tracks(self.tracked_tracks, self.lost_tracks)
        for track in track_pool:
            track.predict()

        dists = iou_distance(track_pool, detections)
        matches, unmatched_tracks, unmatched_detections = self.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            detection = detections[det_idx]
            if track.state == TrackState.Tracked:
                track.update(detection, self.frame_id)
            else:
                track.re_activate(detection, self.frame_id, new_id=False)
            activated_tracks.append(track)

        for track_idx in unmatched_tracks:
            track = track_pool[track_idx]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                lost_tracks.append(track)

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            detection.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(detection)

        self.tracked_tracks = [
            track for track in self.tracked_tracks
            if track.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks = joint_tracks(self.lost_tracks, lost_tracks)
        self._prune_lost_tracks(removed_tracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_tracks = [track for track in self.tracked_tracks if track.is_activated]
        return output_tracks

    def _prune_lost_tracks(self, removed_tracks):
        kept_lost = []
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)
            else:
                kept_lost.append(track)
        self.lost_tracks = kept_lost
        self.removed_tracks = joint_tracks(self.removed_tracks, removed_tracks)

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        try:
            from scipy.optimize import linear_sum_assignment
            matches = linear_sum_assignment(cost_matrix)
            matches = np.asarray(matches).T
            matches = matches[cost_matrix[matches[:, 0], matches[:, 1]] <= thresh]
            unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matches[:, 0]]
            unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matches[:, 1]]
            return matches, unmatched_rows, unmatched_cols
        except ImportError:
            matches = []
            used_rows = set()
            used_cols = set()
            cost_matrix = cost_matrix.copy()
            while True:
                min_val = np.inf
                min_i, min_j = -1, -1
                for i in range(cost_matrix.shape[0]):
                    for j in range(cost_matrix.shape[1]):
                        if i not in used_rows and j not in used_cols and cost_matrix[i, j] < min_val:
                            min_val = cost_matrix[i, j]
                            min_i, min_j = i, j
                if min_val > thresh:
                    break
                matches.append([min_i, min_j])
                used_rows.add(min_i)
                used_cols.add(min_j)
            matches = np.array(matches)
            unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in used_rows]
            unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in used_cols]
            return matches, unmatched_rows, unmatched_cols
