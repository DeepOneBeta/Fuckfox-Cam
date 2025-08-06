# Fuckfox-Cam
基于Luckfox pico mini制作的图传摄像头  
buildroot固件、机械外壳、pcb链接（百度网盘）：  

开发阶段  
第一阶段  
|--硬件部分  
|  |--主控拓展板打样   （已完成）  
|  |--分线板打样       （已完成）  
|  |--尾插小板打样     （已完成）  
|  |--电机驱动板打样   （未完成）  
|  |--外壳绘制         （未完成）  
|  
|--软件部分  
   |--有线串流         （基于luckfox例程已完成）  
   |--系统控制框架     （已完成，具体解释看下文）  
   |--无线串流  
   |--YOLOv5边缘端部署 （毕设，还未开始）  

第二阶段  
|--硬件部分  
|  |--云台控制板打样  
|  |--2W无线功放板打样  
|  
|--软件部分   
   |--云台控制算法开发     
  
第三阶段  
|--待定    
  
# 关于系统框架解释：  
系统运行代码位于：luckfox/App/main.py  
  
系统如果添加自定义代码？  
自定义代码模块路径：luckfox/App/tasks/print_yes.py  
ps:写好的模块可以放在App文件夹下的任何地方，这不是什么要紧的事情，因为在系统注册模块这一步要填写模块路径。  
  
写好的模块代码需要使用的话需要在：luckfox/App/config.json 中进行注册。   
只需要将注册信息写在"tasks"标签下即可。  
大致的书写标准可以参考如下代码例程（此代码为print_yes模块的注册信息）  
{  
"enable": false,  
"module": "tasks.print_yes",  
"class": "PrintYesTask",  
"instance_name": "print_yes_instance",  
"init_args": {  
    "interval": 1.0  
}  
},  

随后在一切完成后运行main.py即可。  
剩下的就祝各位好运。  


