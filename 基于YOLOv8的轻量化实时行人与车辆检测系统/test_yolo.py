from ultralytics import YOLO
import cv2

# 加载官方预训练的 YOLOv8n 模型（自动下载）
model = YOLO('yolov8n.pt')

# 读取一张测试图（可用网络图片或自己拍）
img = cv2.imread('tsst.JPG')

# 推理（只检测 person, car, bus, truck）
results = model(img, classes=[0, 2, 5, 7])  # COCO 类别ID：0=person, 2=car, 5=bus, 7=truck

# 可视化结果
annotated_img = results[0].plot()

# 显示
cv2.imshow('YOLOv8 Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()