# i2c/i2c_stm32_commd.py
import asyncio
import logging
from typing import List
from .i2c_client import I2cClient
from .command_manager import CommandManager
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)

class I2cStm32CommdTask:
    def __init__(
        self,
        config_path: str = "tasks/i2c/i2c_commands.json",
        bus_number: int = 3,
        poll_interval: float = 1.0,
        command: str = "GET_ALL"
    ):
        self.poll_interval = poll_interval
        self.command_name = command

        # 加载配置
        self.cmd_manager = CommandManager(config_path)
        self.client = I2cClient(
            bus_number=bus_number,
            slave_addr=self.cmd_manager.slave_addr,
            response_len=self.cmd_manager.response_len
        )

    async def run(self):
        logger.info("? 启动 I2C 与 STM32 通信任务...")

        # 等待设备上线
        while not self.client.probe():
            logger.warning(f"? STM32 未响应，{self.poll_interval}s 后重试...")
            await asyncio.sleep(self.poll_interval)

        logger.info(f"? 设备已就绪，开始轮询命令: {self.command_name}")

        cmd_byte = self.cmd_manager.get_command(self.command_name)

        while True:
            try:
                data = self.client.send_command(cmd_byte)
                if data is not None:
                    parsed = ResponseParser.parse_csv(data)
                    if parsed:
                        logger.info(f"? 收到数据: {', '.join(parsed)}")
                    else:
                        logger.warning("??  响应解析为空")
                else:
                    logger.error("??  轮询失败，尝试恢复连接...")
                    await self._reconnect()

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info("? I2C 任务被取消")
                break
            except Exception as e:
                logger.critical(f"? 任务异常: {e}")
                break

    async def _reconnect(self):
        """尝试重新连接设备"""
        while True:
            if self.client.probe():
                logger.info("? 设备恢复")
                break
            logger.warning(f"? 重试连接... {self.poll_interval}s 后")
            await asyncio.sleep(self.poll_interval)