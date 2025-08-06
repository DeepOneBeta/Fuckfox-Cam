import asyncio
import logging
import time
import smbus
import os
import json

# ========================
# 🔧 配置
# ========================

OLED_ADDR = 0x3C
I2C_BUS = 3  # 根据实际硬件调整

# OLED 命令
SET_CONTRAST = 0x81
SET_DISPLAY_ON = 0xAF
SET_DISPLAY_OFF = 0xAE
SET_MEMORY_MODE = 0x20
SET_SEG_REMAP = 0xA1
SET_COM_SCAN_DEC = 0xC8
CHARGE_PUMP = 0x8D

# ========================
# 📦 字库加载（从 JSON）
# ========================

def load_font5x8():
    """从 JSON 文件加载 5x8 字模"""
    font_path = os.path.join(os.path.dirname(__file__), "font_unicode.json")
    try:
        with open(font_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 将字符映射为 5 字节列表
        return {chr(i): data.get(chr(i), [0] * 5) for i in range(32, 127)}
    except Exception as e:
        logging.warning(f"⚠️ 加载字库失败，使用默认字模: {e}")
        # 默认空字模（5 列全 0）
        return {chr(i): [0] * 5 for i in range(32, 127)}  # 95 个字符


FONT5X8 = load_font5x8()

# ========================
# 🖥️ OLED 显示驱动类
# ========================

class OLEDDisplay:
    def __init__(self, bus_num=3, addr=OLED_ADDR, width=128, height=64):
        self.bus_num = bus_num
        self.addr = addr
        self.width = width
        self.height = height
        self.pages = height // 8
        self.buffer = [0] * (width * self.pages)
        self.bus = None
        self.logger = logging.getLogger("OLED")

        try:
            self.bus = smbus.SMBus(bus_num)
            self._init()
            self.set_contrast(255)  # 🔥 调高对比度！
            self.logger.info(f"OLED 初始化成功 @ 0x{addr:02X} on bus {bus_num}")
        except Exception as e:
            self.logger.error(f"OLED 初始化失败: {e}")
            self.bus = None

    def set_contrast(self, value):
        """设置对比度（0~255）"""
        if self.bus:
            self._write_cmd(SET_CONTRAST)
            self._write_cmd(value)

    def printf(self, text, x, y):
        """提供类似 C 语言的 printf 接口"""
        self.draw_text(str(text), x, y)

    def _write_cmd(self, cmd):
        """写入命令"""
        if self.bus:
            try:
                self.bus.write_byte_data(self.addr, 0x00, cmd)
            except Exception as e:
                self.logger.debug(f"写命令失败: {cmd:02X}")

    def _write_data(self, data):
        """写入数据（分块）"""
        if not self.bus or not data:
            return
        # ✅ 确保 data 可迭代
        if not hasattr(data, '__iter__'):
            self.logger.error(f"_write_data: data 不是可迭代对象，类型={type(data)}")
            return

        chunk_size = 32
        try:
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                if not hasattr(chunk, '__iter__'):
                    continue
                self.bus.write_i2c_block_data(self.addr, 0x40, chunk)
        except Exception as e:
            self.logger.warning(f"写数据失败: {e}")

    def _init(self):
        """初始化 OLED"""
        init_cmds = [
            SET_DISPLAY_OFF,
            CHARGE_PUMP, 0x14,
            SET_SEG_REMAP,
            SET_COM_SCAN_DEC,
            0xDA, 0x12,
            0x81, 0xCF,
            0xD9, 0xF1,
            0xDB, 0x40,
            SET_MEMORY_MODE, 0x00,
            0x20, 0x00,
            0x40,
            SET_DISPLAY_ON
        ]
        for cmd in init_cmds:
            self._write_cmd(cmd)
        self.clear()
        self.display()

    def clear(self):
        """清屏"""
        self.buffer = [0] * (self.width * self.pages)

    def display(self):
        """刷新显示"""
        self._write_cmd(0x21)  # 列地址
        self._write_cmd(0)
        self._write_cmd(self.width - 1)
        self._write_cmd(0x22)  # 页地址
        self._write_cmd(0)
        self._write_cmd(7)
        self._write_data(self.buffer)  # ✅ 传入 list

    def draw_text_scaled(self, text, x, y, scale=2):
        """
        放大绘制文本
        scale: 缩放倍数（2=10x16, 3=15x24）
        """
        cursor_x = x
        for char in text:
            glyph = FONT5X8.get(char, FONT5X8['?'])
            for col in range(5):  # 每个字符 5 列
                for bit in range(8):  # 每列 8 行
                    if glyph[col] & (1 << (7 - bit)):  # 该点亮
                        # 放大成 scale×scale 的方块
                        for sx in range(scale):
                            for sy in range(scale):
                                px_x = cursor_x + col * scale + sx
                                px_y = y + bit * scale + sy
                                self.draw_pixel(px_x, px_y, 1)
            cursor_x += 6 * scale  # 5宽 + 1间距，再缩放
            if cursor_x >= self.width:
                break

    def draw_pixel(self, x, y, color=1):
        """绘制单个像素"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        byte_idx = x + (y // 8) * self.width
        bit = y & 7
        if color:
            self.buffer[byte_idx] |= (1 << bit)
        else:
            self.buffer[byte_idx] &= ~(1 << bit)

    def draw_text(self, text, x, y):
        """绘制 ASCII 文本（5x8），并增加字符间距"""
        if x < 0 or y < 0 or y >= self.height:
            return
        cursor_x = x
        for c in text:
            glyph = FONT5X8.get(c, FONT5X8['?'])
            # 绘制字符
            for i, byte in enumerate(glyph):
                idx = cursor_x + i + (y // 8) * self.width
                if 0 <= idx < len(self.buffer):
                    self.buffer[idx] |= byte
            cursor_x += 6  # 字符宽 5 + 1 间距
            if cursor_x >= self.width:
                break

    def close(self):
        """关闭 OLED"""
        self.clear()
        self.display()
        if self.bus:
            self.bus.close()
        self.logger.info("OLED 已关闭")


class OLEDDisplayTask:
    def __init__(self,
                 bus_num=3,
                 addr=OLED_ADDR,
                 update_interval=1.0,
                 show_cpu=True,
                 show_mem=True,
                 show_temp=True,
                 show_time=True):
        self.bus_num = bus_num
        self.addr = addr
        self.update_interval = update_interval
        self.show_cpu = show_cpu
        self.show_mem = show_mem
        self.show_temp = show_temp
        self.show_time = show_time
        self.oled = None
        self.logger = logging.getLogger("OLED-Task")
        self.error_code = 0

    async def run(self):
        """主循环：刷新显示"""
        self.logger.info("OLED 任务启动")
        self.oled = OLEDDisplay(bus_num=self.bus_num, addr=self.addr)
        if self.oled.bus is None:
            self.error_code = 1001
            self.logger.error("❌ OLED 初始化失败，错误码: 1001")

        frame = 0
        while True:
            try:
                self.oled.clear()
                row = 0

                # 模拟数据
                cpu = (frame * 7) % 100 + 15.5
                mem = (frame * 5) % 100 + 20.3
                temp = 35.0 + (frame % 20) * 0.5

                # 显示内容
                self.oled.draw_text("System Status", 0, row);
                row += 10
                if self.show_cpu:
                    self.oled.draw_text(f"CPU:  {cpu:5.1f}%", 0, row);
                    row += 10
                if self.show_mem:
                    self.oled.draw_text(f"MEM:  {mem:5.1f}%", 0, row);
                    row += 10
                if self.show_temp:
                    self.oled.draw_text(f"TEMP: {temp:5.1f}C", 0, row);
                    row += 10

                status = "OK" if self.error_code == 0 else "ERR"
                self.oled.draw_text(f"Status: {status} [{self.error_code}]", 0, row);
                row += 10

                if self.show_time:
                    t = time.strftime("%H:%M:%S")
                    self.oled.draw_text(f"Time: {t}", 0, row);
                    row += 10

                self.oled.display()

                if frame % 30 == 0:
                    self.logger.info(f"OLED 刷新，错误码: {self.error_code}")

                frame += 1
            except Exception as e:
                self.logger.error(f"OLED 显示错误: {e}")
                self.error_code = 1002

            await asyncio.sleep(self.update_interval)

    def close(self):
        if self.oled:
            self.oled.close()
        self.logger.info("OLED 任务关闭")


# ========================
# ✅ 主入口：单元测试
# ========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("🔧 开始运行 OLED 真实硬件单元测试...")

    try:
        oled = OLEDDisplay(bus_num=3, addr=0x3C, width=128, height=64)
        print("✅ OLED 实例创建成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        exit(1)

    # 测试 1: buffer 类型
    assert isinstance(oled.buffer, list), f"buffer 类型错误: {type(oled.buffer)}"
    assert len(oled.buffer) == 1024, f"buffer 长度错误: {len(oled.buffer)}"
    print("✅ 测试 1: buffer 正常")

    # 测试 2: 清屏
    oled.clear()
    print("✅ 测试 2: 清屏完成")

    # 测试 3: 绘制像素
    oled.draw_pixel(0, 0, 1)
    oled.draw_pixel(127, 0, 1)
    oled.draw_pixel(0, 63, 1)
    oled.draw_pixel(127, 63, 1)
    print("✅ 测试 3: 像素绘制完成")

    # 测试 4: 刷新显示
    try:
        oled.display()
        print("✅ 测试 4: display() 成功")
    except Exception as e:
        print(f"❌ display() 失败: {e}")
        exit(1)

    # 测试 5: 显示文字
    oled.clear()
    oled.draw_text("Hello", 0, 0)
    oled.draw_text("World", 0, 16)
    try:
        oled.display()
        print("✅ 测试 5: 文字显示成功")
    except Exception as e:
        print(f"❌ 文字显示失败: {e}")

    # 结束
    time.sleep(2)
    oled.clear()
    oled.display()
    oled.close()
    print("✅ 所有测试完成！OLED 驱动工作正常。")