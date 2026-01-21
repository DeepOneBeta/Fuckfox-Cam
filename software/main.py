import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
from collections import deque
import time

# ===== 调试开关 =====
DEBUG = True
DEBUG_PRINT_INTERVAL_SEC = 1.0
_last_debug_print_ts = 0.0

# ===== 疲劳检测参数（优化版） =====
EAR_THRESHOLD_BASE = 0.21          # 基础眼睛闭合阈值（将自适应调整）
MOUTH_RATIO_THRESHOLD_BASE = 0.32  # 基础嘴巴开合阈值
CONSEC_FRAMES_EYE = 15             # 连续闭眼帧数
CONSEC_FRAMES_MOUTH = 25           # 连续张嘴帧数

# ===== 优化参数 =====
SMOOTH_WINDOW_SIZE = 5             # 滑动窗口大小（平滑处理）
ADAPTIVE_THRESHOLD_ALPHA = 0.1     # 自适应阈值更新系数（0-1，越小越稳定）
PERCLOS_WINDOW_SECONDS = 60        # PERCLOS时间窗口（秒）
BLINK_DURATION_THRESHOLD = 0.2     # 眨眼持续时间阈值（秒）
MIN_BLINK_INTERVAL = 0.5           # 最小眨眼间隔（秒，避免重复计数）
HEAD_POSE_THRESHOLD = 30           # 头部姿态角度阈值（度，超过此值视为无效）

# ===== 状态变量 =====
eye_close_count = 0
mouth_open_count = 0
fatigue_alert = False

# ===== 优化数据结构 =====
ear_history = deque(maxlen=SMOOTH_WINDOW_SIZE)      # EAR历史值（滑动窗口）
mouth_history = deque(maxlen=SMOOTH_WINDOW_SIZE)    # 嘴巴比例历史值
ear_baseline = None                                  # EAR基线值（自适应阈值）
mouth_baseline = None                                # 嘴巴基线值
blink_times = deque(maxlen=100)                     # 眨眼时间戳（用于PERCLOS）
last_blink_time = 0                                  # 上次眨眼时间
blink_start_time = None                              # 当前眨眼开始时间
perclos_window_start = time.time()                   # PERCLOS窗口开始时间

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

def calculate_head_pose(landmarks):
    """
    计算头部姿态角度（pitch, yaw）
    返回角度（度），用于过滤无效帧
    使用鼻尖、下巴、左眼角、右眼角等关键点
    """
    if len(landmarks) < 468:  # MediaPipe Face Landmarker有468个关键点
        return 0.0  # 关键点不足，返回0（视为有效）
    
    # MediaPipe关键点索引（基于468点模型）
    # 鼻尖: 4, 下巴: 175, 左眼角: 33, 右眼角: 263
    try:
        nose_tip = landmarks[4]
        chin = landmarks[175] if len(landmarks) > 175 else landmarks[17]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # 计算pitch（上下点头）- 使用鼻尖到下巴的向量
        nose_chin_vec = (chin[0] - nose_tip[0], chin[1] - nose_tip[1])
        pitch = math.degrees(math.atan2(abs(nose_chin_vec[1]), abs(nose_chin_vec[0])))
        
        # 计算yaw（左右转头）- 使用双眼连线
        eye_vec = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        eye_distance = math.sqrt(eye_vec[0]**2 + eye_vec[1]**2)
        if eye_distance > 0:
            # 计算眼睛连线的角度
            yaw = math.degrees(math.atan2(abs(eye_vec[1]), abs(eye_vec[0])))
        else:
            yaw = 0.0
        
        return max(pitch, yaw)  # 返回最大角度
    except (IndexError, TypeError):
        return 0.0  # 出错时返回0（视为有效，避免误判）

def smooth_value(value, history, window_size=SMOOTH_WINDOW_SIZE):
    """
    滑动窗口平滑处理
    使用移动平均减少噪声
    """
    history.append(value)
    if len(history) < window_size:
        return value  # 窗口未满时直接返回
    return np.mean(list(history))

def update_adaptive_threshold(current_value, baseline, alpha=ADAPTIVE_THRESHOLD_ALPHA):
    """
    更新自适应阈值
    使用指数移动平均（EMA）动态调整基线
    """
    if baseline is None:
        return current_value
    return alpha * current_value + (1 - alpha) * baseline

def calculate_perclos(blink_times, window_start_time, window_duration=PERCLOS_WINDOW_SECONDS):
    """
    计算PERCLOS（Percentage of Eyelid Closure）
    在时间窗口内眼睛闭合的时间百分比
    """
    current_time = time.time()
    if current_time - window_start_time < window_duration:
        return 0.0
    
    # 统计窗口内的眨眼次数和持续时间
    window_start = current_time - window_duration
    valid_blinks = [bt for bt in blink_times if bt >= window_start]
    
    if len(valid_blinks) < 2:
        return 0.0
    
    # 简化版PERCLOS：基于眨眼频率
    blink_frequency = len(valid_blinks) / window_duration
    # 正常眨眼频率：15-20次/分钟，低于10次/分钟可能疲劳
    normal_blink_rate = 0.25  # 次/秒（15次/分钟）
    if blink_frequency < normal_blink_rate * 0.5:  # 低于正常值50%
        return 1.0 - (blink_frequency / normal_blink_rate)
    return 0.0

def detect_blink(ear_value, threshold, current_time):
    """
    检测眨眼事件
    返回是否检测到眨眼
    """
    global last_blink_time, blink_start_time
    
    is_closed = ear_value < threshold
    
    if is_closed:
        if blink_start_time is None:
            blink_start_time = current_time
    else:
        if blink_start_time is not None:
            # 眼睛睁开，检查是否是一个完整的眨眼
            blink_duration = current_time - blink_start_time
            time_since_last = current_time - last_blink_time
            
            # 眨眼持续时间合理，且距离上次眨眼足够久
            if (BLINK_DURATION_THRESHOLD * 0.1 <= blink_duration <= BLINK_DURATION_THRESHOLD * 2 and
                time_since_last >= MIN_BLINK_INTERVAL):
                blink_times.append(current_time)
                last_blink_time = current_time
                blink_start_time = None
                return True
            blink_start_time = None
    
    return False

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
    head_pose_valid = True
    head_pose_angle = 0.0
    current_time = time.time()

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            landmarks = []
            for lm in face_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.append((x, y))

            # 检查头部姿态（只影响疲劳判断，不影响特征点绘制）
            head_pose_angle = calculate_head_pose(landmarks)
            head_pose_valid = head_pose_angle < HEAD_POSE_THRESHOLD

            # 提取关键区域
            left_eye = [landmarks[i] for i in LEFT_EYE_IDXS]
            right_eye = [landmarks[i] for i in RIGHT_EYE_IDXS]
            mouth_for_ratio = [landmarks[i] for i in MOUTH_RATIO_IDXS]
            mouth_for_draw = [landmarks[i] for i in MOUTH_CONTOUR_IDXS]

            # 计算指标
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            raw_ear = (left_ear + right_ear) / 2.0
            raw_mouth_ratio = calculate_mouth_ratio(mouth_for_ratio)

            # 滑动窗口平滑处理
            current_ear = smooth_value(raw_ear, ear_history)
            current_mouth_ratio = smooth_value(raw_mouth_ratio, mouth_history)

            # 更新自适应阈值（基线）
            if ear_baseline is None:
                ear_baseline = current_ear
            else:
                # 只在眼睛睁开时更新基线
                if current_ear > ear_baseline * 0.9:  # 避免闭眼时更新
                    ear_baseline = update_adaptive_threshold(current_ear, ear_baseline)

            if mouth_baseline is None:
                mouth_baseline = current_mouth_ratio
            else:
                # 只在嘴巴闭合时更新基线
                if current_mouth_ratio < mouth_baseline * 1.1:  # 避免张嘴时更新
                    mouth_baseline = update_adaptive_threshold(current_mouth_ratio, mouth_baseline)

            # 使用自适应阈值
            adaptive_ear_threshold = ear_baseline * 0.7 if ear_baseline else EAR_THRESHOLD_BASE
            adaptive_mouth_threshold = mouth_baseline * 1.5 if mouth_baseline else MOUTH_RATIO_THRESHOLD_BASE

            # 检测眨眼
            detect_blink(current_ear, adaptive_ear_threshold, current_time)

            # 绘制
            cv2.polylines(annotated_frame, [np.array(left_eye, np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(annotated_frame, [np.array(right_eye, np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(annotated_frame, [np.array(mouth_for_draw, np.int32)], True, (0, 0, 255), 2)

            face_found = True
            # 只处理第一张脸（num_faces=1），避免重复覆盖指标/绘制
            break
    else:
        face_found = False

    # ===== 优化后的疲劳检测逻辑 =====
    perclos_score = 0.0
    fatigue_score = 0.0
    
    if face_found and head_pose_valid:
        # 计算PERCLOS
        perclos_score = calculate_perclos(blink_times, perclos_window_start)
        
        # 使用自适应阈值的闭眼检测
        adaptive_ear_threshold = ear_baseline * 0.7 if ear_baseline else EAR_THRESHOLD_BASE
        if current_ear < adaptive_ear_threshold:
            eye_close_count += 1
        else:
            eye_close_count = max(0, eye_close_count - 2)

        # 使用自适应阈值的张嘴检测
        adaptive_mouth_threshold = mouth_baseline * 1.5 if mouth_baseline else MOUTH_RATIO_THRESHOLD_BASE
        if current_mouth_ratio > adaptive_mouth_threshold:
            mouth_open_count += 1
        else:
            mouth_open_count = max(0, mouth_open_count - 3)

        # 多指标融合评分
        eye_score = min(eye_close_count / CONSEC_FRAMES_EYE, 1.0)
        mouth_score = min(mouth_open_count / CONSEC_FRAMES_MOUTH, 1.0)
        perclos_weight = 0.3  # PERCLOS权重
        eye_weight = 0.5       # 闭眼权重
        mouth_weight = 0.2     # 张嘴权重
        
        fatigue_score = (eye_weight * eye_score + 
                        mouth_weight * mouth_score + 
                        perclos_weight * perclos_score)

        # 触发警告（使用综合评分）
        if fatigue_score >= 0.7 or eye_close_count >= CONSEC_FRAMES_EYE or mouth_open_count >= CONSEC_FRAMES_MOUTH:
            fatigue_alert = True
        else:
            fatigue_alert = False
    else:
        # 未检测到人脸或头部姿态不佳时，缓慢重置计数器
        eye_close_count = max(0, eye_close_count - 1)
        mouth_open_count = max(0, mouth_open_count - 1)
        if not face_found:
            fatigue_alert = False

    # ===== 显示信息（优化版） =====
    y_offset = 30
    line_height = 25

    # 基础状态（总是显示，便于定位是否检测到脸）
    cv2.putText(
        annotated_frame,
        f"Face: {'YES' if face_found else 'NO'}",
        (10, y_offset + line_height * 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if face_found else (0, 0, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"PoseAngle: {head_pose_angle:.1f} deg",
        (10, y_offset + line_height * 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if head_pose_valid else (0, 0, 255),
        2,
    )
    
    # 基础指标
    cv2.putText(annotated_frame, f"EAR: {current_ear:.3f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 自适应阈值显示
    if ear_baseline is not None:
        adaptive_thresh = ear_baseline * 0.7
        cv2.putText(annotated_frame, f"EAR Threshold: {adaptive_thresh:.3f} (Adaptive)", (10, y_offset + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.putText(annotated_frame, f"Mouth Ratio: {current_mouth_ratio:.3f}", (10, y_offset + line_height * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 计数器
    cv2.putText(annotated_frame, f"Eye Close: {eye_close_count}/{CONSEC_FRAMES_EYE}", (10, y_offset + line_height * 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Mouth Open: {mouth_open_count}/{CONSEC_FRAMES_MOUTH}", (10, y_offset + line_height * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # PERCLOS和综合评分
    if face_found and head_pose_valid:
        cv2.putText(annotated_frame, f"PERCLOS: {perclos_score:.2f}", (10, y_offset + line_height * 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(annotated_frame, f"Fatigue Score: {fatigue_score:.2f}", (10, y_offset + line_height * 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(annotated_frame, f"Blinks: {len(blink_times)}", (10, y_offset + line_height * 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 头部姿态状态
    if face_found:
        status_color = (0, 255, 0) if head_pose_valid else (0, 0, 255)
        status_text = "Head Pose: OK" if head_pose_valid else "Head Pose: Invalid"
        cv2.putText(annotated_frame, status_text, (10, y_offset + line_height * 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # 疲劳警告（更醒目的显示）
    if fatigue_alert:
        # 半透明红色背景
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 100), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "⚠️ FATIGUE ALERT! TAKE A BREAK!", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('Fatigue Detection - RTSP (Robust)', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ===== 终端调试输出（节流，避免刷屏） =====
    if DEBUG:
        global _last_debug_print_ts
        if current_time - _last_debug_print_ts >= DEBUG_PRINT_INTERVAL_SEC:
            _last_debug_print_ts = current_time
            num_faces = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
            adaptive_ear_threshold = ear_baseline * 0.7 if ear_baseline else EAR_THRESHOLD_BASE
            adaptive_mouth_threshold = mouth_baseline * 1.5 if mouth_baseline else MOUTH_RATIO_THRESHOLD_BASE
            print(
                f"[debug] faces={num_faces} face_found={face_found} "
                f"pose_angle={head_pose_angle:.1f} valid={head_pose_valid} "
                f"EAR={current_ear:.3f} thr={adaptive_ear_threshold:.3f} "
                f"Mouth={current_mouth_ratio:.3f} thr={adaptive_mouth_threshold:.3f} "
                f"score={fatigue_score:.2f} alert={fatigue_alert}"
            )

# ===== 清理 =====
landmarker.close()
cap.release()
cv2.destroyAllWindows()