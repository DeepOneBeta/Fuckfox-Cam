import asyncio
import logging

logger = logging.getLogger(__name__)

class PrintYesTask:
    """
    简单打印任务，定期输出"yes"
    配置参数：
      interval: 打印间隔(秒)
      prefix: 可选前缀文本
    """
    
    def __init__(self, interval: float = 1.0, prefix: str = ""):
        self.interval = interval
        self.prefix = prefix
        self._stop_event = asyncio.Event()
        
        # 验证参数有效性
        if interval <= 0:
            raise ValueError("间隔时间必须大于0")
        
        logger.info(f"✅ PrintYesTask初始化 | 间隔: {interval}s | 前缀: '{prefix}'")

    async def run(self):
        """主协程任务循环"""
        logger.info("🟢 PrintYesTask任务启动")
        counter = 0
        
        try:
            while not self._stop_event.is_set():
                # 核心打印逻辑
                counter += 1
                output = f"{self.prefix}yes" if self.prefix else "yes"
                print(f"#{counter} -> {output}")
                
                # 带异常捕获的等待
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), 
                        timeout=self.interval
                    )
                except asyncio.TimeoutError:
                    continue
                
        except asyncio.CancelledError:
            logger.warning("🟡 任务被取消")
        finally:
            logger.info("🔴 PrintYesTask任务终止")

    def stop(self):
        """安全停止任务"""
        self._stop_event.set()
        logger.debug("🛑 收到停止信号")
