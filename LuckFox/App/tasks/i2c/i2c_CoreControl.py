# tasks/i2c/i2c_stm32_comm.py

import asyncio
import smbus
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

class I2cStm32CommdTask:
    def __init__(
        self,
        config_path: str = "tasks/i2c/i2c_commands.json",
        bus_number: int = 3,
        poll_interval: float = 1.0,
        command: str = "GET_ALL"
    ):
        """
        初始化 I2C 与 STM32 通信任务
        :param config_path: 指令配置 JSON 文件路径
        :param bus_number: I2C 总线号
        :param poll_interval: 轮询间隔（秒）
        :param command: 要发送的命令名称
        """
        self.bus_number = bus_number
        self.poll_interval = poll_interval
        self.command = command
        self.bus = None

        # 加载配置
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"❌ I2C 配置文件未找到: {config_path}")

        self.commands: Dict[str, int] = {}
        self.slave_addr: int = 0
        self.response_len: int = 10

        self._load_config()

        if command not in self.commands:
            raise ValueError(f"❌ 未知的 I2C 命令: {command}")

    def _load_config(self):
        """加载 JSON 配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.commands = data.get("commands", {})
            self.slave_addr = data.get("slave_addr", 0)
            self.response_len = data.get("response_len", 10)

            logger.info(f"✅ 成功加载 I2C 配置: {self.config_path}")
            logger.debug(f"🔧 命令集: {self.commands}")
            logger.debug(f"📍 从机地址: 0x{self.slave_addr:02X}")
        except Exception as e:
            logger.critical(f"❌ 加载 I2C 配置失败: {e}")
            raise

    def probe_slave(self) -> bool:
        """探测从机是否存在"""
        try:
            bus = smbus.SMBus(self.bus_number)
            bus.write_byte(self.slave_addr, 0x00)
            bus.close()
            logger.info(f"✅ STM32 设备在地址 0x{self.slave_addr:02X} 响应成功")
            return True
        except Exception as e:
            logger.error(f"❌ 未在 0x{self.slave_addr:02X} 发现 STM32 设备: {e}")
            return False

    def send_command_and_read(self) -> Optional[List[int]]:
        """发送命令并读取响应"""
        for retry in range(3):
            try:
                self.bus = smbus.SMBus(self.bus_number)
                cmd_byte = self.commands[self.command]
                self.bus.write_byte(self.slave_addr, cmd_byte)
                logger.debug(f"📤 发送命令: {self.command} (0x{cmd_byte:02X})")

                data = self.bus.read_i2c_block_data(self.slave_addr, 0x00, self.response_len)
                logger.debug(f"📥 原始响应: {[f'0x{x:02X}' for x in data]}")
                self.bus.close()
                return data
            except Exception as e:
                logger.warning(f"🔁 第 {retry+1} 次通信失败: {e}")
                if self.bus:
                    self.bus.close()
                asyncio.sleep(0.1)
        logger.error(f"❌ 发送 '{self.command}' 失败，重试 3 次均无响应")
        return None

    @staticmethod
    def parse_csv_response(data: List[int]) -> List[str]:
        """解析字节流为字符串字段（CSV 格式）"""
        try:
            raw_bytes = bytes(data)
            if 0 in raw_bytes:
                raw_bytes = raw_bytes[:raw_bytes.index(0)]
            text = raw_bytes.decode('utf-8').strip()
            fields = [f.strip() for f in text.split(',') if f.strip()]
            return fields
        except Exception as e:
            logger.error(f"❌ 解析响应失败: {e}")
            return []

    async def run(self):
        """协程主循环"""
        logger.info("🚀 启动 I2C 与 STM32 通信任务...")

        # 初始探测 & 等待设备上线
        while True:
            if self.probe_slave():
                logger.info(f"✅ STM32 设备已就绪，开始轮询命令: {self.command}")
                break
            else:
                logger.warning(f"🛑 STM32 设备未响应，将在 {self.poll_interval:.1f}s 后重试...")
                await asyncio.sleep(self.poll_interval)  # 等待后重试

        # 主循环：持续轮询
        try:
            while True:
                response = self.send_command_and_read()
                if response is not None:
                    parsed = self.parse_csv_response(response)
                    if parsed:
                        logger.info(f"📊 收到数据: {', '.join(parsed)}")
                    else:
                        logger.warning("⚠️  响应解析为空或格式错误")
                else:
                    logger.error("⚠️  本次轮询失败，设备可能掉线")

                    # 进入等待恢复模式
                    logger.info("🔁 尝试重新连接设备...")
                    while True:
                        if self.probe_slave():
                            logger.info("✅ 设备恢复在线，继续轮询")
                            break
                        logger.warning(f"🔧 设备仍不在线，{self.poll_interval:.1f}s 后重试...")
                        await asyncio.sleep(self.poll_interval)

                # 正常轮询间隔
                await asyncio.sleep(self.poll_interval)

        except asyncio.CancelledError:
            logger.info("🛑 I2C 任务被取消")
        except Exception as e:
            logger.critical(f"💥 I2C 任务异常终止: {e}")
        finally:
            if self.bus:
                try:
                    self.bus.close()
                except:
                    pass