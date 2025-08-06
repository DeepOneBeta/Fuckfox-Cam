# Fuckfox-Cam
---

基于Luckfox pico mini制作的图传摄像头  
buildroot固件、机械外壳、pcb链接（百度网盘）：  

---

##  项目开发阶段

###  第一阶段：基础功能搭建

####  硬件部分

| 组件 | 状态 |
| ---- | ---- |
| 主控拓展板打样 | ✅ 已完成 |
| 分线板打样 | ✅ 已完成 |
| 尾插小板打样 | ✅ 已完成 |
| 电机驱动板打样 | ⏳ 未完成 |
| 外壳绘制 | ⏳ 未完成 |

####  软件部分

| 功能 | 状态 |
| ---- | ---- |
| 有线串流（基于 LuckFox 例程） | ✅ 已完成 |
| 系统控制框架 | ✅ 已完成，具体解释看下文 |
| 无线串流 | ⏳ 开发中 |
| YOLOv5 边缘端部署（毕设） | ⏳ 未开始 |

---

###  第二阶段：云台与无线增强

####  硬件部分

| 组件 | 状态 |
| ---- | ---- |
| 云台控制板打样 | ⏳ 待完成 |
| 2W 无线功放板打样 | ⏳ 待完成 |

#### 💻 软件部分

| 功能 | 状态 |
| ---- | ---- |
| 云台控制算法开发 | ⏳ 待完成 |

---

###  第三阶段：系统集成与优化

| 部分 | 描述 |
| ---- | ---- |
| 硬件部分 | 待定 |
| 软件部分 | 待定 |

---

#  LuckFox 系统框架使用说明

本系统基于 Python `asyncio` 构建，采用模块化设计，支持通过配置文件灵活注册和管理自定义任务模块。适用于嵌入式设备（如 LuckFox）上的多任务协同控制场景。

---

##  项目结构
luckfox/
└── App/
├── main.py               # 系统主程序入口
├── config.json           # 任务配置文件（注册自定义模块）
└── tasks/
└── print_yes.py          # 示例任务模块（可扩展）

---

##  如何添加自定义任务？

只需三步，即可将你的功能接入系统。

### 1. 编写任务模块

将你的功能封装为一个 Python 类，建议存放于 `App/tasks/` 目录下（路径可自定义）。  

示例：`App/tasks/print_yes.py`
```python
import asyncio

class PrintYesTask:
    def __init__(self, interval: float = 1.0):
        self.interval = interval

    async def run(self):
        while True:
            print("Yes")
            await asyncio.sleep(self.interval)
```

要求：  
类必须实现 async def run(self) 方法，作为协程运行入口。  
所有初始化参数通过 __init__ 接收，并在配置中传入。  

2. 在 config.json 中注册模块  
编辑 App/config.json，在 "tasks" 数组中添加注册信息：  

```json
{
  "tasks": [
    {
      "enable": true,
      "module": "tasks.print_yes",
      "class": "PrintYesTask",
      "instance_name": "print_yes_instance",
      "init_args": {
        "interval": 1.0
      }
    }
  ]
}
```

字段说明  
| 字段名 | 说明 |
| ---- | ---- |
| enable | 是否启用该任务（true/false） |
| module | Python 模块路径（如 tasks.oled.display） | 
| class | 要实例化的类名 | 
| instance_name | 实例名称，用于日志标识和调试 | 
| init_args | 传递给类构造函数的参数（支持任意 JSON 类型） | 

ps：模块路径需与文件实际位置匹配，例如 tasks/i2c/stm32_comm 对应 App/tasks/i2c/stm32_comm.py  

3. 启动系统  
运行主程序，系统将自动加载并运行所有已启用的任务：  
python App/main.py  

⚠️ 注意事项
所有任务必须提供 async def run(self) 方法。  
避免使用阻塞调用（如 time.sleep()），应使用 await asyncio.sleep()。  
若需调用阻塞函数（如 smbus、串口、文件读写等），请使用 loop.run_in_executor 包装，防止阻塞事件循环。  

祝你开发顺利，好运常伴！🍀

—— D.D.D  

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！  
本项目适合嵌入式、物联网、边缘计算方向的开发者参与。  

---
