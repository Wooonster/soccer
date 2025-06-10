#!/bin/bash

# 足球项目开发启动脚本

echo "正在启动足球项目开发环境..."

# 检查SSH隧道是否已存在
if lsof -Pi :50001 -sTCP:LISTEN -t >/dev/null ; then
    echo "端口 50001 已被占用，可能SSH隧道已建立"
else
    echo "建立SSH隧道到远程服务器..."
    # 后台建立SSH隧道
    ssh -p 14984 -L 50001:localhost:50001 -f -N root@connect.cqa1.seetacloud.com
    
    if [ $? -eq 0 ]; then
        echo "SSH隧道建立成功"
    else
        echo "SSH隧道建立失败，请检查网络连接和SSH配置"
        exit 1
    fi
fi

# 等待隧道建立
sleep 2

# 测试后端连接
echo "测试后端连接..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:50001/api/videos 2>/dev/null)

if [ "$response" = "200" ]; then
    echo "后端连接正常"
else
    echo "警告: 后端连接可能有问题 (HTTP状态码: $response)"
    echo "请确保远程服务器上的Flask应用正在运行"
fi

# 进入前端目录
cd soccer-vue

# 安装依赖（如果需要）
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

# 启动前端开发服务器
echo "启动前端开发服务器..."
npm run dev 