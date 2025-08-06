# main.py
"""
LuckFox RV1103 协程主程序
功能：从 config.json 读取任务配置，动态导入并启动协程
"""

import asyncio
import logging
import json
import importlib
from types import SimpleNamespace
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | \033[1m%(name)-12s\033[0m | %(levelname)8s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = None
        self.instances = {}
        self.tasks: List[asyncio.Task] = []

    async def load_config(self):
        """加载 JSON 配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"✅ 配置文件 '{self.config_file}' 加载成功")
        except Exception as e:
            logger.critical(f"❌ 无法加载配置文件: {e}")
            raise

    def instantiate_tasks(self):
        """根据配置动态实例化任务"""
        task_configs = self.config.get("tasks", [])
        if not task_configs:
            logger.warning("⚠️  配置中没有定义任务")
            return

        for cfg in task_configs:
            if not cfg.get("enable", False):
                logger.info(f"⏩ 跳过未启用任务: {cfg.get('class', 'Unknown')}")
                continue

            module_name = cfg["module"]
            class_name = cfg["class"]
            instance_name = cfg.get("instance_name", class_name.lower())
            init_args = cfg.get("init_args", {})

            try:
                # 动态导入模块
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)

                # 创建实例
                instance = cls(**init_args)
                self.instances[instance_name] = instance

                # 检查是否有 run() 协程方法
                if not hasattr(instance, "run") or not asyncio.iscoroutinefunction(getattr(instance, "run")):
                    logger.error(f"❌ {class_name} 缺少 async run() 方法")
                    continue

                # 创建协程任务
                task = asyncio.create_task(instance.run(), name=class_name)
                self.tasks.append(task)
                logger.info(f"✅ 已启动任务: {class_name} ({instance_name})")

            except Exception as e:
                logger.error(f"❌ 实例化 {class_name} 失败: {e}")

    async def run(self):
        """运行所有任务"""
        await self.load_config()
        self.instantiate_tasks()

        if not self.tasks:
            logger.warning("⚠️  没有可运行的任务，程序退出")
            return

        logger.info(f"🚀 开始运行 {len(self.tasks)} 个协程任务...")

        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("🛑 用户中断")
        except Exception as e:
            logger.error(f"❌ 主任务异常: {e}")
        finally:
            logger.info("🧹 正在取消所有任务...")
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            logger.info("✅ 所有任务已关闭")


# ================================
#        程序入口
# ================================
if __name__ == "__main__":
    logger.info("🚀 启动 LuckFox RV1103 动态协程控制系统...")

    # 你可以把 test_simulator.py 改名为 tasks/ 下的文件
    # 或者直接修改 config.json 中的 module 路径为 "test_simulator"
    # 例如: "module": "test_simulator"

    task_manager = TaskManager("config.json")

    try:
        asyncio.run(task_manager.run())
    except KeyboardInterrupt:
        print("\n👋 程序已退出。")
    except Exception as e:
        logger.critical(f"💥 启动失败: {e}")
        exit(1)