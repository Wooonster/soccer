#!/bin/bash

# 足球项目开发环境停止脚本

echo "正在停止足球项目开发环境..."

# 查找并终止SSH隧道进程
echo "查找SSH隧道进程..."
ssh_pids=$(ps aux | grep "ssh.*-L 50001:localhost:50001" | grep -v grep | awk '{print $2}')

if [ -n "$ssh_pids" ]; then
    echo "发现SSH隧道进程: $ssh_pids"
    echo "正在终止SSH隧道..."
    echo $ssh_pids | xargs kill
    
    if [ $? -eq 0 ]; then
        echo "SSH隧道已停止"
    else
        echo "停止SSH隧道时出现错误"
    fi
else
    echo "未发现活跃的SSH隧道进程"
fi

# 检查端口是否释放
sleep 1
if lsof -Pi :50001 -sTCP:LISTEN -t >/dev/null ; then
    echo "警告: 端口 50001 仍被占用"
    echo "可能需要手动终止相关进程:"
    lsof -Pi :50001 -sTCP:LISTEN
else
    echo "端口 50001 已释放"
fi

echo "开发环境停止完成" 