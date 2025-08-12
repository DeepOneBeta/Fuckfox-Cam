#!/bin/bash

# I2C bus number and device address
I2C_BUS=3
OLED_ADDR=0x3c

# SSD1306 initialization commands
commands=(
    0xAE        # 关闭显示
    0xD5 0x80   # 设置时钟分频因子
    0xA8 0x3F   # 设置多路复用率 (1/64 duty)
    0xD3 0x00   # 设置显示偏移（无偏移）
    0x40        # 设置显示起始行
    0x8D 0x14   # 启用充电泵
    0x20 0x00   # 设置内存寻址模式为水平模式
    0xA1        # 设置段重映射（A0/A1）
    0xC8        # 设置 COM 输出扫描方向
    0xDA 0x12   # 设置 COM 引脚硬件配置
    0x81 0xCF   # 设置对比度
    0xD9 0xF1   # 设置预充电周期
    0xDB 0x40   # 设置 VCOMH 取消选择级别
    0xA4        # 启用全屏点亮（非全开模式）
    0xA6        # 设置正常显示（非反色）
    0x21 0x00 0x7F  # 设置列地址（0-127）
    0x22 0x00 0x07  # 设置页地址（0-7）
    0xAF        # 开启显示
)

echo "Initializing OLED at address $OLED_ADDR on bus $I2C_BUS..."

for cmd in "${commands[@]}"; do
    if [[ ${#cmd} -eq 2 ]]; then
        i2cset -y $I2C_BUS $OLED_ADDR 0x00 $cmd
    else
        i2cset -y $I2C_BUS $OLED_ADDR 0x40 $cmd
    fi
done

echo "Initialization complete."



