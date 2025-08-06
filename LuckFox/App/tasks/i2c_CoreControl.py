import smbus
import time
import sys

# --- 指令字典 (与STM32端必须一致!) ---
COMMANDS = {
    "GET_TEMP": 0x01,
    "GET_HUMI": 0x02,
    "GET_PRESS": 0x03,
    "GET_ALL":  0x04,
    "SET_LED":  0x10,
    "RESET":    0xFF,
}

# --- 配置 ---
I2C_BUS = 3             # 根据实际环境修改，对应 /dev/i2c-3
SLAVE_ADDR = 0x24       # 必须与STM32一致
RESPONSE_LEN = 10       # 与STM32的 tx_csv_buffer[10] 一致
MAX_RETRY = 3           # 通信失败重试次数
DELAY_BETWEEN = 0.1     # 命令间隔（秒）

def probe_slave_address(bus_number: int, target_addr: int) -> bool:
    """探测从机是否存在：仅写一个字节，看是否 ACK"""
    bus = None
    try:
        bus = smbus.SMBus(bus_number)
        bus.write_byte(target_addr, 0x00)  # 发送一个字节，不读
        print(f"? Device ACKed at address 0x{target_addr:02X}")
        return True
    except Exception as e:
        print(f"? No device found at 0x{target_addr:02X}: {e}")
        return False
    finally:
        if bus:
            bus.close()

def send_command_and_read_response(bus_number: int, slave_addr: int, cmd_name: str) -> list:
    """
    发送命令并读取响应
    使用 write_byte + read_i2c_block_data
    注意：不要在没有发送命令时调用 read！
    """
    bus = None
    for retry in range(MAX_RETRY):
        try:
            bus = smbus.SMBus(bus_number)

            # 1. 发送命令（Write）
            cmd_byte = COMMANDS[cmd_name]
            bus.write_byte(slave_addr, cmd_byte)
            print(f"??  Sent '{cmd_name}' (0x{cmd_byte:02X})")

            # 2. 立即读取响应（Read）—— 关键：紧跟在 write 之后
            data = bus.read_i2c_block_data(slave_addr, 0x00, RESPONSE_LEN)
            print(f"   Raw response: {[f'0x{x:02X}' for x in data]}")

            bus.close()
            return data

        except Exception as e:
            print(f"   ??  Attempt {retry+1} failed: {e}")
            time.sleep(0.1)
            if bus:
                bus.close()
        finally:
            pass

    print(f"? Failed to get response for '{cmd_name}' after {MAX_RETRY} retries.")
    return None

def parse_csv_response(data: list) -> list:
    """解析字节流为字符串字段（以 \0 或逗号分割）"""
    try:
        # 转为 bytes，去掉尾部 0
        raw_bytes = bytes(data)
        if 0 in raw_bytes:
            raw_bytes = raw_bytes[:raw_bytes.index(0)]
        text = raw_bytes.decode('utf-8').strip()
        fields = [f.strip() for f in text.split(',') if f.strip()]
        return fields
    except Exception as e:
        print(f"? Parse error: {e}")
        return []

def main():
    print("=== I2C Slave Test (Using ONLY smbus, NO smbus2) ===")
    print("??  Warning: Never read from I2C without sending a command first!")

    # 1. 探测设备
    if not probe_slave_address(I2C_BUS, SLAVE_ADDR):
        print("? Slave not found. Check wiring, address, power, or I2C bus number.")
        sys.exit(1)

    # 2. 测试命令
    test_commands = ["GET_TEMP", "GET_HUMI", "GET_PRESS", "GET_ALL"]

    for cmd in test_commands:
        print(f"\n--- Testing: {cmd} ---")
        response = send_command_and_read_response(I2C_BUS, SLAVE_ADDR, cmd)
        if response is not None:
            fields = parse_csv_response(response)
            print("   Parsed values:")
            for i, val in enumerate(fields):
                print(f"     [{i}]: {val}")
        time.sleep(DELAY_BETWEEN)

    print("\n? All tests completed.")

if __name__ == "__main__":
    main()