import cv2
import numpy as np
import time

class YOLOv8Detector:
    def __init__(self, onnx_model, input_size=640, conf_thres=0.5, iou_thres=0.45):
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(onnx_model)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.target_class_ids = [0, 2, 5, 7]
        # 完整的类别名称映射
        self.class_names = {
            0: 'person',
            2: 'car',
            5: 'bus',
            7: 'truck'
        }

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # 创建带灰边的 640x640 canvas
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dw, dh = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        canvas[dh:dh + new_h, dw:dw + new_w, :] = resized

        # 生成 blob
        blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        return blob, scale, dw, dh
    def postprocess(self, outputs, img_shape, scale, dw, dh):
        outputs = np.squeeze(outputs).T  # (8400, 84)
        boxes, scores, class_ids = [], [], []
        h_img, w_img = img_shape[:2]
        input_size = self.input_size

        for det in outputs:
            x, y, w, h = det[:4]
            if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
                x *= input_size
                y *= input_size
                w *= input_size
                h *= input_size
            class_scores = det[4:]
            if class_scores.max(initial=0) > 1 or class_scores.min(initial=0) < 0:
                class_scores = 1 / (1 + np.exp(-class_scores))
            class_id = np.argmax(class_scores)
            max_conf = class_scores[class_id]
            if max_conf < self.conf_thres or class_id not in self.target_class_ids:
                continue

            # 坐标去掉 padding 偏移，并映射回原图
            x1 = (x - w / 2 - dw) / scale
            y1 = (y - h / 2 - dh) / scale
            x2 = (x + w / 2 - dw) / scale
            y2 = (y + h / 2 - dh) / scale

            # 边界裁剪
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img - 1, int(x2)), min(h_img - 1, int(y2))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(max_conf))
            class_ids.append(int(class_id))

        if len(boxes) == 0:
            return [], [], []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        final_boxes, final_scores, final_classes = [], [], []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                final_boxes.append([x, y, x + w, y + h])
                final_scores.append(scores[i])
                final_classes.append(class_ids[i])

        return final_boxes, final_scores, final_classes

    def detect(self, img):
        blob, scale, dw, dh = self.preprocess(img)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self.postprocess(outputs, img.shape, scale, dw, dh)


# 测试
if __name__ == "__main__":
    detector = YOLOv8Detector("yolov8n.onnx")
    cap = cv2.VideoCapture(0)  # 0=摄像头，或替换为 "video.mp4"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        boxes, scores, class_ids = detector.detect(frame)
        fps = 1 / (time.time() - start_time)

        if len(boxes) == 0:
            cv2.putText(frame, "No objects detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


        # 绘制结果
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{detector.class_names.get(cls_id, f'ID{cls_id}')} {score:.2f}"
            print(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("YOLOv8 ONNX Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
