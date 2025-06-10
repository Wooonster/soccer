# SSH隧道连接指南

## 方法1: 本地端口转发（推荐）

在本地终端中运行以下命令，建立SSH隧道：

```bash
ssh -p 14984 -L 50001:localhost:50001 root@connect.cqa1.seetacloud.com
```

这个命令的含义：
- `-L 50001:localhost:50001`: 将本地的50001端口转发到远程服务器的50001端口
- `-p 14984`: 指定SSH端口
- `root@connect.cqa1.seetacloud.com`: 远程服务器地址

建立隧道后，保持此终端窗口打开。

## 方法2: 后台运行SSH隧道

```bash
ssh -p 14984 -L 50001:localhost:50001 -f -N root@connect.cqa1.seetacloud.com
```

参数说明：
- `-f`: 后台运行
- `-N`: 不执行远程命令，只做端口转发

## 方法3: 使用autossh（自动重连）

如果需要自动重连功能：

```bash
autossh -p 14984 -M 0 -L 50001:localhost:50001 -f -N root@connect.cqa1.seetacloud.com
```

## 验证连接

隧道建立后，您可以通过以下方式测试：

```bash
curl http://localhost:50001/api/videos
```

## 注意事项

1. 确保远程服务器上的Flask应用正在运行
2. 确保Flask应用绑定到了正确的地址（0.0.0.0:50001）
3. 保持SSH隧道连接开启状态
4. 如果遇到权限问题，确保SSH密钥配置正确 