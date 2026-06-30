# utils/visualizer.py
import cv2

def draw_tracks(frame, tracked_objects, show_id=True, show_bbox=True):
    output_frame = frame.copy()
    h, w = frame.shape[:2]
    
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['bbox']
        track_id = obj['id']
        class_id = obj['class_id']
        
        # 边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if show_bbox:
            # 不同类别的框用不同颜色
            color = (0, 255, 0)  # 默认绿色
            if class_id == 1:   # cup
                color = (255, 0, 0)  # 蓝色
            elif class_id == 2: # chair
                color = (0, 0, 255)  # 红色
            elif class_id == 5: # phone
                color = (255, 255, 0) # 青色
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        if show_id:
            # 显示 "类别名 ID:1"
            class_names = {
                0: 'person',
                1: 'bicycle',
                2: 'car',          # ← 修正这里！
                3: 'motorcycle',
                5: 'bus',
                7: 'truck'
            }.get(class_id, f'cls{class_id}')
            label = f"{class_names} ID:{track_id}"
            cv2.putText(
                output_frame, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )
    return output_frame

def draw_alerts(frame, alerts):
    output_frame = frame.copy()
    for alert in alerts:
        x, y = alert['location']
        if alert['type'] == 'crowd':
            text = f"CROWD! ({alert['count']} people)"
            color = (0, 0, 255)
        elif alert['type'] == 'fall':
            text = "FALL DETECTED!"
            color = (0, 0, 255)
        elif alert['type'] == 'loitering':
            text = "LOITERING!"
            color = (255, 0, 0)
        
        cv2.putText(output_frame, text, (x, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.circle(output_frame, (x, y), 10, color, -1)
    return output_frame