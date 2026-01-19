import torch
import cv2
import numpy as np
import os
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def load_model(weights='yolov5s.pt', device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names = model.stride, model.names
    model.model.float()
    return model, device, stride, names


# ✅ 修复1: 添加 device 参数
def preprocess_frame(frame, device, img_size=640, stride=32):
    img = letterbox(frame, img_size, stride=stride)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


if __name__ == '__main__':
    weights = r'F:\Fuckfox-Cam\software\yolov5\runs\train\fatigue_yolov5n\weights\best.pt'

    # ✅ 检查文件是否存在
    if not os.path.exists(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    img_size = 640
    conf_thres = 0.25  # ✅ 先调低测试
    iou_thres = 0.45
    device = ''

    model, device, stride, names = load_model(weights, device=device)
    print(f"模型加载成功！检测类别: {names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("无法打开摄像头")

    print("按 'q' 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break

        # ✅ 修复2: 传入 device
        img = preprocess_frame(frame, device, img_size=img_size, stride=stride)

        with torch.no_grad():
            pred = model(img, augment=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=100)

        # ✅ 调试：打印检测数量
        for i, det in enumerate(pred):
            print(f"第 {i} 帧检测到 {len(det)} 个目标")
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('YOLOv5 Camera Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")