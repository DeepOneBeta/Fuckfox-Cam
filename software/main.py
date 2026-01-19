import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

# ===== 疲劳检测参数 =====
EAR_THRESHOLD = 0.21          # 眼睛闭合阈值
MOUTH_RATIO_THRESHOLD = 0.32   # 嘴巴开合阈值（建议先观察再调）
CONSEC_FRAMES_EYE = 15        # 连续闭眼帧数
CONSEC_FRAMES_MOUTH = 25      # 连续张嘴帧数

eye_close_count = 0
mouth_open_count = 0
fatigue_alert = False

# ===== 关键点索引 =====
# 👁️ 眼睛（各6点）
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

# 👄 嘴巴（仅4个最稳定点用于开合度计算）✅
MOUTH_RATIO_IDXS = [61, 291, 0, 17]  # 左嘴角, 右嘴角, 鼻下点, 下巴顶点

# 🖌️ 嘴巴轮廓（20点，仅绘图用）
MOUTH_CONTOUR_IDXS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]

# ===== 初始化 FaceLandmarker =====
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ===== 打开 RTSP 流 =====
rtsp_url = "rtsp://172.32.0.93/live/0"
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少延迟

print(f"连接 RTSP: {rtsp_url}")
print("按 'q' 退出")

def calculate_ear(eye):
    """计算眼睛纵横比 EAR"""
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def calculate_mouth_ratio(mouth_pts):
    """
    计算嘴巴开合度（更鲁棒）
    mouth_pts: [left_corner, right_corner, upper_lip_center, lower_lip_center]
    使用垂直像素距离 / 嘴角宽度
    """
    left, right, upper, lower = mouth_pts
    # 仅使用 y 坐标差（避免旋转/姿态影响）
    mouth_height = abs(lower[1] - upper[1])
    mouth_width = math.dist(left, right)
    return mouth_height / mouth_width if mouth_width > 1e-5 else 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取 RTSP 流，请检查网络或摄像头")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = landmarker.detect(mp_image)
    annotated_frame = frame.copy()

    current_ear = 0.0
    current_mouth_ratio = 0.0
    face_found = False

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            landmarks = []
            for lm in face_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.append((x, y))

            # 提取关键区域
            left_eye = [landmarks[i] for i in LEFT_EYE_IDXS]
            right_eye = [landmarks[i] for i in RIGHT_EYE_IDXS]
            mouth_for_ratio = [landmarks[i] for i in MOUTH_RATIO_IDXS]
            mouth_for_draw = [landmarks[i] for i in MOUTH_CONTOUR_IDXS]

            # 计算指标
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            current_ear = (left_ear + right_ear) / 2.0
            current_mouth_ratio = calculate_mouth_ratio(mouth_for_ratio)

            # 绘制
            cv2.polylines(annotated_frame, [np.array(left_eye, np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(annotated_frame, [np.array(right_eye, np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(annotated_frame, [np.array(mouth_for_draw, np.int32)], True, (0, 0, 255), 2)

            face_found = True

    # ===== 疲劳逻辑 =====
    if face_found:
        # 闭眼检测
        if current_ear < EAR_THRESHOLD:
            eye_close_count += 1
        else:
            eye_close_count = max(0, eye_close_count - 2)

        # 张嘴检测（使用新 ratio）
        if current_mouth_ratio > MOUTH_RATIO_THRESHOLD:
            mouth_open_count += 1
        else:
            mouth_open_count = max(0, mouth_open_count - 3)

        # 触发警告
        if eye_close_count >= CONSEC_FRAMES_EYE or mouth_open_count >= CONSEC_FRAMES_MOUTH:
            fatigue_alert = True
        else:
            fatigue_alert = False
    else:
        eye_close_count = 0
        mouth_open_count = 0
        fatigue_alert = False

    # ===== 显示信息 =====
    cv2.putText(annotated_frame, f"EAR: {current_ear:.3f}", (10, 30+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Mouth Ratio: {current_mouth_ratio:.3f}", (10, 60+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Eye Close: {eye_close_count}/{CONSEC_FRAMES_EYE}", (10, 90+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Mouth Open: {mouth_open_count}/{CONSEC_FRAMES_MOUTH}", (10, 115+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if fatigue_alert:
        cv2.putText(annotated_frame, "⚠️ FATIGUE ALERT! TAKE A BREAK!", (50, 200+130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow('Fatigue Detection - RTSP (Robust)', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== 清理 =====
landmarker.close()
cap.release()
cv2.destroyAllWindows()