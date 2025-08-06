# tasks/print_yes.py

import asyncio
import logging

logger = logging.getLogger(__name__)

class PrintYesTask:
    def __init__(self, interval: float = 1.0):
        """
        初始化任务
        :param interval: 每隔多少秒打印一次 "yes"
        """
        self.interval = interval

    async def run(self):
        """协程主函数：循环打印 yes"""
        logger.info("? PrintYesTask 已启动，开始打印 'yes'")
        try:
            while True:
                print("yes")
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            logger.info("? PrintYesTask 被取消")
        except Exception as e:
            logger.error(f"? PrintYesTask 出现异常: {e}")
            