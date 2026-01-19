# train_fatigue_yolov5.py
import os
import sys
from pathlib import Path

# 确保在 yolov5 目录下运行，或添加路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'  # 修改为你的 yolov5 源码路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train import run

if __name__ == '__main__':
    # 配置训练参数
    opt = {
        'imgsz': 640,                     # 输入图像尺寸
        'batch_size': 30,                 # 批大小（根据显存调整）
        'epochs': 90,                    # 训练轮数
        'data': r'F:\Fuckfox-Cam\software\fatigue_driving\data.yaml',  # 数据集配置文件
        'weights': 'yolov5s.pt',          # 预训练权重（确保在 yolov5 目录下）
        'cfg': 'models/yolov5s.yaml',     # 模型配置
        'name': 'fatigue_yolov5s',        # 实验名称（结果保存在 runs/train/fatigue_yolov5n）
        'device': '0',                    # GPU 设备（'cpu' 或 '0,1' 多卡）
        'cache': False,                   # 是否缓存图像到内存（大显存可开）
        'workers': 6,                     # 数据加载线程数
        'project': 'runs/train',          # 项目目录
        'exist_ok': False,                # 如果实验名存在是否覆盖
        'quad': False,
        'rect': False,
        'resume': False,
        'nosave': False,
        'noval': False,
        'noautoanchor': False,
        'noplots': False,
        'evolve': None,
        'bucket': '',
        'save_period': -1,
        'artifact_alias': 'latest',
        'local_rank': -1,
        'freeze': [0],                    # 冻结前 n 层（可选）
        'optimizer': 'SGD',               # 优化器
        'cos_lr': False,
        'label_smoothing': 0.0,
        'patience': 100,                  # EarlyStop patience（设很大=禁用）
    }

    # 启动训练
    print("🚀 开始训练疲劳驾驶检测模型...")
    run(**opt)
    print("✅ 训练完成！模型保存在: runs/train/fatigue_yolov5n/weights/best.pt")