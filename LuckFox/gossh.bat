@echo off
:: 设置代码页为 UTF-8
chcp 65001 > nul

cls
echo.
echo ================================
echo   正在连接到 MPU 开发板
echo   IP: 172.32.0.93
echo   用户名: root
echo   密码: luckfox (已复制到剪贴板，请粘贴)
echo ================================
echo.

:: 将密码复制到剪贴板
echo luckfox| clip

:: 开始 SSH 连接
echo 正在启动 SSH 连接...
echo (请在密码提示处按 Ctrl+V 粘贴密码)
echo.
ssh root@172.32.0.93

echo.
echo 连接已断开。
pause