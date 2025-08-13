# i2c/response_parser.py
from typing import List, Optional

class ResponseParser:
    @staticmethod
    def parse_csv(data: List[int]) -> List[str]:
        """解析字节流为 CSV 字符串列表"""
        try:
            raw_bytes = bytes(data)
            # 截断到第一个空字节
            if 0 in raw_bytes:
                raw_bytes = raw_bytes[:raw_bytes.index(0)]
            text = raw_bytes.decode('utf-8').strip()
            return [f.strip() for f in text.split(',') if f.strip()]
        except Exception as e:
            print(f"解析失败: {e}")
            return []

    @staticmethod
    def parse_binary(data: List[int]) -> bytes:
        """返回原始字节"""
        return bytes(data)