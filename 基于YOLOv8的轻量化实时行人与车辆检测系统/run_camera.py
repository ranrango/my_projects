import cv2
import numpy as np
import time
from detector import YOLOv8Detector  # 假设你的检测器类在 detector.py 中

def main():
    # 初始化检测器（确保 yolov8n.onnx 在同目录）
    detector = YOLOv8Detector("yolov8n.onnx")
    
    # 打开摄像头（0 表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 可选：设置摄像头分辨率（提升FPS）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    print("🚀 按 'q' 退出程序")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频帧")
            break

        # 记录推理开始时间
        start_time = time.time()
        
        # 执行检测
        boxes, scores, class_ids = detector.detect(frame)
        
        # 计算FPS
        fps = 1.0 / (time.time() - start_time)
        
        # 绘制检测结果
        for box, score, cls_id in zip(boxes, scores, class_ids):
            # 关键：转为 Python 原生 int（避免OpenCV报错）
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # 边界保护（防止坐标越界）
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 绘制绿色框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{detector.class_names.get(cls_id, 'unknown')} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow("YOLOv8 Light Detector - Press 'q' to quit", frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 程序已退出")

if __name__ == "__main__":
    main()
