# i2c/command_manager.py
import json
from pathlib import Path
from typing import Dict, Any

class CommandManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        self.config = self._load_config()
        self.commands: Dict[str, int] = self.config.get("commands", {})
        self.slave_addr: int = self.config.get("slave_addr", 0)
        self.response_len: int = self.config.get("response_len", 10)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_command(self, name: str) -> int:
        if name not in self.commands:
            raise ValueError(f"未知命令: {name}")
        return self.commands[name]