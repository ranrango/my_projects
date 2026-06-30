import cv2
import numpy as np
import time
import argparse
from detector import YOLOv8Detector

def main():
    # 命令行参数：支持指定视频文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='', help='输出视频路径（可选）')
    parser.add_argument('--max-frames', type=int, default=0, help='最多处理帧数，0 表示处理完整视频')
    parser.add_argument('--no-display', action='store_true', help='不弹出 OpenCV 窗口，适合命令行验收')
    args = parser.parse_args()

    # 初始化检测器
    detector = YOLOv8Detector("yolov8n.onnx")
    
    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {args.video}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 视频信息: {width}x{height} @ {fps} FPS")

    # 视频写入器（如果指定了输出路径）
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("🚀 按 'q' 退出播放")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ 视频播放完毕")
            break

        # 记录推理时间
        start_time = time.time()
        boxes, scores, class_ids = detector.detect(frame)
        infer_time = time.time() - start_time

        # 绘制结果
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{detector.class_names.get(cls_id, f'ID{cls_id}')} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示FPS和推理时间
        cv2.putText(frame, f"FPS: {1/infer_time:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 写入输出视频
        if out:
            out.write(frame)

        # 显示画面
        if not args.no_display:
            cv2.imshow("YOLOv8 Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.max_frames and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= args.max_frames:
            print(f"✅ 已处理 {args.max_frames} 帧")
            break

    # 释放资源
    cap.release()
    if out:
        out.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("✅ 程序已退出")

if __name__ == "__main__":
    main()
