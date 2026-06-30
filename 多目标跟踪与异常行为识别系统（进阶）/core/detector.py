# core/detector.py
import cv2
import numpy as np

class YOLOv8Detector:
    def __init__(
        self,
        onnx_model,
        input_size=640,
        conf_thres=0.5,
        iou_thres=0.45,
        target_class_ids=None,
        verbose=False,
    ):
        self.input_size = input_size
        self.net = cv2.dnn.readNetFromONNX(onnx_model)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.verbose = verbose
        self.target_class_ids = target_class_ids or [0, 1, 2, 3, 5, 7]
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))
        pad_top = (self.input_size - new_h) // 2
        pad_left = (self.input_size - new_w) // 2

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        return blob, scale, pad_left, pad_top, (h, w)

    def postprocess(self, outputs, scale, pad_left, pad_top, img_shape):
        outputs = np.squeeze(outputs)
        if outputs.shape[0] == 84:
            outputs = outputs.T

        h_img, w_img = img_shape
        boxes, scores, class_ids = [], [], []

        for det in outputs:
            x_center, y_center, w, h = det[:4]
            if max(abs(x_center), abs(y_center), abs(w), abs(h)) <= 1.5:
                x_center *= self.input_size
                y_center *= self.input_size
                w *= self.input_size
                h *= self.input_size

            cls_scores = det[4:84]
            if cls_scores.max(initial=0) > 1 or cls_scores.min(initial=0) < 0:
                cls_scores = 1 / (1 + np.exp(-cls_scores))
            cls_id = np.argmax(cls_scores)
            score = cls_scores[cls_id]

            if score < self.conf_thres or cls_id not in self.target_class_ids:
                continue

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # 去除 padding 并还原比例
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale

            # 边界裁剪
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(0, min(w_img - 1, x2))
            y2 = max(0, min(h_img - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(int(cls_id))

        if len(boxes) == 0:
            return [], [], []

        # NMS
        nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf_thres, self.iou_thres)

        if len(indices) == 0:
            return [], [], []

        if isinstance(indices, tuple):
            indices = indices[0]
        elif hasattr(indices, 'flatten'):
            indices = indices.flatten()

        final_boxes = [boxes[i] for i in indices]
        final_scores = [scores[i] for i in indices]
        final_classes = [class_ids[i] for i in indices]

        return final_boxes, final_scores, final_classes

    def detect(self, img):
        blob, scale, pad_left, pad_top, img_shape = self.preprocess(img)
        if self.verbose:
            print(f"[DEBUG] scale={scale:.3f}, pad_left={pad_left}, pad_top={pad_top}, orig={img_shape}")

        self.net.setInput(blob)
        outputs = self.net.forward()
        if self.verbose:
            print(f"[DEBUG] Raw output shape: {outputs.shape}")

        boxes, scores, classes = self.postprocess(outputs, scale, pad_left, pad_top, img_shape)
        if self.verbose:
            print(f"[INFO] Detections: {len(boxes)} objects")
        return boxes, scores, classes
