# App/tasks/test_motor.py
import asyncio
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-12s | %(levelname)8s | %(message)s'
)

class MotorControl:
    def __init__(self, motor_id="M1", speed=50, direction="forward"):
        self.motor_id = motor_id
        self.speed = speed
        self.direction = direction
        self.logger = logging.getLogger(f"Motor-{motor_id}")

    async def run(self):
        """模拟电机控制协程：周期性打印状态"""
        self.logger.info(f"🟢 电机 {self.motor_id} 启动 | 方向: {self.direction}, 速度: {self.speed}%")
        
        while True:
            self.logger.info(f"🔁 电机 {self.motor_id} 正常运行中...")
            await asyncio.sleep(2.0)  # 模拟每 2 秒一次控制循环