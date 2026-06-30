# anomaly/anomaly_engine.py
import math
from collections import defaultdict, deque

class AnomalyEngine:
    def __init__(self, 
                 fps=30,
                 person_class_id=0,
                 crowd_enabled=True,
                 crowd_threshold=3,
                 crowd_radius=100,
                 crowd_duration=2,
                 fall_enabled=True,
                 fall_speed_thresh=15,
                 fall_ratio_thresh=0.7,
                 loitering_enabled=True,
                 loitering_area=50,
                 loitering_duration=5,
                 alert_cooldown=5):
        self.trajectories = defaultdict(lambda: deque(maxlen=200))
        self.active_alerts = {}
        self.last_alert_frame = {}
        self.frame_count = 0
        self.fps = fps
        self.person_class_id = person_class_id
        
        self.crowd_enabled = crowd_enabled
        self.crowd_threshold = crowd_threshold
        self.crowd_radius = crowd_radius
        self.crowd_min_frames = int(crowd_duration * self.fps)
        self.fall_enabled = fall_enabled
        self.fall_speed_thresh = fall_speed_thresh
        self.fall_ratio_thresh = fall_ratio_thresh
        self.loitering_enabled = loitering_enabled
        self.loitering_area = loitering_area
        self.loitering_min_frames = int(loitering_duration * self.fps)
        self.alert_cooldown_frames = int(alert_cooldown * self.fps)

    def update(self, tracked_objects, frame_id):
        self.frame_count = frame_id
        current_time = frame_id
        people = [obj for obj in tracked_objects if obj.get('class_id') == self.person_class_id]
        
        active_ids = set()
        for obj in people:
            tid = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            w, h = x2 - x1, y2 - y1
            center = obj['center']
            self.trajectories[tid].append((center[0], center[1], w, h, current_time))
            active_ids.add(tid)
        
        for tid in list(self.trajectories.keys()):
            if tid not in active_ids:
                self.trajectories.pop(tid, None)
                for key in list(self.active_alerts.keys()):
                    if f"loitering_{tid}" in key:
                        del self.active_alerts[key]
        
        new_alerts = []
        if self.crowd_enabled:
            new_alerts.extend(self._detect_crowd(people, current_time))
        if self.fall_enabled:
            new_alerts.extend(self._detect_fall(people, current_time))
        if self.loitering_enabled:
            new_alerts.extend(self._detect_loitering(current_time))
        
        return new_alerts

    def _cooldown_ready(self, key, current_time):
        last_frame = self.last_alert_frame.get(key)
        if last_frame is None or current_time - last_frame >= self.alert_cooldown_frames:
            self.last_alert_frame[key] = current_time
            return True
        return False

    def _detect_crowd(self, objects, current_time):
        if len(objects) < self.crowd_threshold:
            return []
        
        alerts = []
        for i, obj1 in enumerate(objects):
            count = 1
            neighbors = [obj1['center']]
            for j, obj2 in enumerate(objects):
                if i != j:
                    dist = math.hypot(
                        obj1['center'][0] - obj2['center'][0],
                        obj1['center'][1] - obj2['center'][1]
                    )
                    if dist < self.crowd_radius:
                        count += 1
                        neighbors.append(obj2['center'])
            
            if count >= self.crowd_threshold:
                alert_key = f"crowd_{min(obj['id'] for obj in objects if obj['center'] in neighbors)}"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = current_time
                else:
                    duration = current_time - self.active_alerts[alert_key]
                    if duration >= self.crowd_min_frames:
                        avg_x = sum(p[0] for p in neighbors) / len(neighbors)
                        avg_y = sum(p[1] for p in neighbors) / len(neighbors)
                        if self._cooldown_ready(alert_key, current_time):
                            alerts.append({
                                'type': 'crowd',
                                'location': (int(avg_x), int(avg_y)),
                                'count': count,
                                'frame_id': current_time
                            })
                        self.active_alerts[alert_key] = current_time - self.crowd_min_frames + 15
                break
        return alerts

    def _detect_fall(self, objects, current_time):
        alerts = []
        for obj in objects:
            tid = obj['id']
            traj = self.trajectories[tid]
            if len(traj) < 5:
                continue
            
            prev = traj[-2]
            curr = traj[-1]
            x_prev, y_prev, w_prev, h_prev, t_prev = prev
            x_curr, y_curr, w_curr, h_curr, t_curr = curr
            
            ratio_prev = h_prev / (w_prev + 1e-6)
            ratio_curr = h_curr / (w_curr + 1e-6)
            if ratio_prev > 1.5 and ratio_curr < self.fall_ratio_thresh:
                dy = y_curr - y_prev
                dt = t_curr - t_prev
                speed_y = dy / (dt + 1e-6)
                if speed_y > self.fall_speed_thresh:
                    alert_key = f"fall_{tid}"
                    if self._cooldown_ready(alert_key, current_time):
                        alerts.append({
                            'type': 'fall',
                            'location': obj['center'],
                            'track_id': tid,
                            'frame_id': current_time
                        })
        return alerts

    def _detect_loitering(self, current_time):
        alerts = []
        for tid, traj in self.trajectories.items():
            if len(traj) < self.loitering_min_frames:
                continue
            
            recent_traj = list(traj)[-self.loitering_min_frames:]
            xs = [p[0] for p in recent_traj]
            ys = [p[1] for p in recent_traj]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            area_width = x_max - x_min
            area_height = y_max - y_min
            
            if area_width < self.loitering_area and area_height < self.loitering_area:
                alert_key = f"loitering_{tid}"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = current_time
                else:
                    duration = current_time - self.active_alerts[alert_key]
                    if duration >= self.loitering_min_frames:
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        if self._cooldown_ready(alert_key, current_time):
                            alerts.append({
                                'type': 'loitering',
                                'location': (center_x, center_y),
                                'track_id': tid,
                                'frame_id': current_time
                            })
                        self.active_alerts[alert_key] = current_time - self.loitering_min_frames + 30
        return alerts
