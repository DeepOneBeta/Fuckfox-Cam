#!/bin/bash

# 用于从网络服务获取北京时间的URL
TIME_URL="http://worldtimeapi.org/api/timezone/Asia/Shanghai"

# 尝试获取北京时间
response=$(curl -s $TIME_URL)

if [[ "$response" == *"error"* ]]; then
    echo "无法从网络服务获取时间"
    exit 1
fi

# 解析响应中的时间信息
datetime=$(echo $response | grep -oP '"datetime":.*?[^\\]",?' | cut -d '"' -f 4)
datetime=${datetime%?} # 去除末尾逗号

if [ -z "$datetime" ]; then
    echo "解析时间失败"
    exit 1
fi

# 转换为Unix时间戳
timestamp=$(date -d "$datetime" +%s)

if [ $? -ne 0 ]; then
    echo "转换时间戳失败"
    exit 1
fi

# 设置系统时间为获取到的时间
if command -v date >/dev/null 2>&1; then
    sudo date +%s -s @$timestamp
    if [ $? -eq 0 ]; then
        echo "系统时间已更新为: $datetime"
    else
        echo "设置系统时间失败"
        exit 1
    fi
else
    echo "找不到'date'命令，请检查您的系统环境"
    exit 1
fi

exit 0