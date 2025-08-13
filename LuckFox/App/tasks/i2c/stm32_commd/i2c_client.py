# i2c/i2c_client.py
import smbus
import logging
import asyncio

logger = logging.getLogger(__name__)

class I2cClient:
    def __init__(self, bus_number: int, slave_addr: int, response_len: int = 10):
        self.bus_number = bus_number
        self.slave_addr = slave_addr
        self.response_len = response_len
        self.bus = None

    def probe(self) -> bool:
        """探测从机是否在线"""
        try:
            bus = smbus.SMBus(self.bus_number)
            bus.write_byte(self.slave_addr, 0x00)
            bus.close()
            logger.info(f"✅ I2C 设备 0x{self.slave_addr:02X} 在线")
            return True
        except Exception as e:
            logger.error(f"❌ I2C 设备 0x{self.slave_addr:02X} 无响应: {e}")
            return False

    def send_command(self, cmd_byte: int) -> list:
        """发送命令并读取响应"""
        for retry in range(3):
            try:
                self.bus = smbus.SMBus(self.bus_number)
                self.bus.write_byte(self.slave_addr, cmd_byte)
                logger.debug(f"📤 发送命令字节: 0x{cmd_byte:02X}")

                data = self.bus.read_i2c_block_data(self.slave_addr, 0x00, self.response_len)
                logger.debug(f"📥 接收到数据: {[f'0x{x:02X}' for x in data]}")
                return data
            except Exception as e:
                logger.warning(f"🔁 第 {retry+1} 次通信失败: {e}")
                if self.bus:
                    self.bus.close()
                asyncio.sleep(0.1)
        logger.error("❌ 重试3次后仍无法通信")
        return None