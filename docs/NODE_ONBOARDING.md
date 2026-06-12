# DecentralAI 节点加入指南 v0.1

## 概述

节点加入网络需要：下载二进制 → 配置身份 → 启动服务 → 验证连接。

---

## 硬件分级（决定角色）

| 级别 | 硬件要求 | 模型能力 | 月收入预估 |
|------|----------|----------|------------|
| **L0** | 任意 CPU | 数据采集 | 贡献积分 |
| **L1** | CPU + 4GB RAM | 0.5~1.5B 参数 | ¥5-15 |
| **L2** | RTX 3060 8GB | 7B (Q4 量化) | ¥50-200 |
| **L3** | RTX 3090/4090 | 14B+ (Q4 量化) | ¥200-500 |
| **L4** | A100/H100 | 70B 骨干 | ¥500-2000 |

---

## 第一步：下载/构建二进制

### 方式 A：从 release 下载（推荐）
```bash
# Windows
curl -LO https://github.com/YOUR_USERNAME/decentral-ai/releases/latest/decentral-ai-core.exe

# Linux/macOS
curl -LO https://github.com/YOUR_USERNAME/decentral-ai/releases/latest/decentral-ai-core
chmod +x decentral-ai-core
```

### 方式 B：从源码构建
```bash
# 安装 Rust（如果还没装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 克隆并构建
git clone https://github.com/YOUR_USERNAME/decentral-ai.git
cd decentral-ai/src-rs/decentral-ai-core
cargo build --release

# 二进制位置: target/release/decentral-ai-core
```

---

## 第二步：创建配置文件

在节点机器上创建 `config.yaml`：

```yaml
# 节点身份
node:
  id: "node-$(hostname)-$(date +%s)"  # 自动生成唯一ID
  role: "L2"                            # 根据硬件选择级别
  alias: "我的3060矿机"                  # 可选，方便识别

# 网络配置
network:
  api_port: 8080         # HTTP API 端口
  p2p_port: 9090         # P2P 通信端口
  bootstrap_nodes:        # 引导节点（启动时连接）
    - "1.2.3.4:9090"     # 替换为实际引导节点地址
    - "5.6.7.8:9090"

# 区块链配置（用于结算）
blockchain:
  rpc_url: "https://rpc.example.com"  # 替换为实际 RPC
  wallet_private_key: ""               # 留空则自动生成新钱包
  contract_address: "0x..."            # DecentralAI 合约地址

# 推理配置（L1+ 节点）
inference:
  model_path: "models/rwkv-4-169m-native.pth"  # 模型文件路径
  max_batch_size: 4                             # 最大并发请求数
  gpu_enabled: true                              # 是否使用 GPU
```

---

## 第三步：启动节点

### 基本启动
```bash
./decentral-ai-core

# 指定角色
./decentral-ai-core node-abc123 L2

# 带配置文件
./decentral-ai-core --config config.yaml
```

### Docker 部署（推荐用于服务器）
```dockerfile
FROM rust:1.94 as builder
WORKDIR /app
COPY src-rs/decentral-ai-core ./src-rs/decentral-ai-core
RUN cargo build --release --manifest-path=src-rs/decentral-ai-core/Cargo.toml

FROM debian:bookworm-slim
COPY --from=builder /app/src-rs/decentral-ai-core/target/release/decentral-ai-core /usr/local/bin/
COPY config.yaml /etc/decentral-ai/config.yaml
EXPOSE 8080 9090
CMD ["decentral-ai-core", "--config", "/etc/decentral-ai/config.yaml"]
```

```bash
# 构建并运行
docker build -t decentral-ai-node .
docker run -d \
  --name decentral-ai-node \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /path/to/models:/models \
  decentral-ai-node
```

### Systemd 服务（Linux 服务器）
```ini
# /etc/systemd/system/decentral-ai.service
[Unit]
Description=DecentralAI Node
After=network.target

[Service]
Type=simple
User=decentral
WorkingDirectory=/home/decentral
ExecStart=/home/decentral/decentral-ai-core --config /home/decentral/config.yaml
Restart=on-failure
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable decentral-ai
sudo systemctl start decentral-ai
sudo systemctl status decentral-ai
```

---

## 第四步：验证节点运行

```bash
# 健康检查
curl http://localhost:8080/health
# 期望: {"status":"healthy","service":"decentral-ai-core","version":"0.1.0"}

# 查看节点信息
curl http://localhost:8080/v1/models
# 期望: {"id":"node-xxx","role":"L2","capabilities":["inference"],...}

# 查看信用余额
curl http://localhost:8080/credits
# 期望: {"node_id":"xxx","balance":100.0,"reputation":50.0}

# P2P 连接状态（未来版本）
curl http://localhost:8080/v1/peers
```

---

## 第五步：加入 P2P 网络（v0.2+）

当 P2P 网络上线后，节点会自动：

1. **引导阶段**：连接到 bootstrap_nodes 列表中的节点
2. **发现阶段**：通过 gossipsub 广播自己的存在
3. **同步阶段**：接收其他节点的路由信息

```
[新节点] → [Bootstrap Node] → [整个网络]
    ↓
  接收邻居列表
    ↓
  加入 gossipsub 话题
```

---

## 收益机制

节点通过以下方式赚取 Credits：

| 行为 | 奖励 | 说明 |
|------|------|------|
| 处理推理请求 | 1.5 Credits/请求 | 根据模型大小和难度浮动 |
| 提供训练数据 | 0.5 Credits/样本 | 需经审核 |
| 保持在线 | 0.1 Credits/小时 | 可靠性奖励 |
| 被投诉验证失败 | -5 Credits | 质量惩罚 |

---

## 常见问题

**Q: 防火墙需要开放哪些端口？**
```powershell
# Windows
New-NetFirewallRule -DisplayName "DecentralAI API" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "DecentralAI P2P" -Direction Inbound -LocalPort 9090 -Protocol TCP -Action Allow
```

**Q: 节点能同时跑多个吗？**
可以，每个节点用不同 ID 和端口：
```bash
./decentral-ai-core node-001 L2 --api-port 8080 --p2p-port 9090
./decentral-ai-core node-002 L1 --api-port 8081 --p2p-port 9091
```

**Q: 节点会被 DOS 攻击吗？**
有速率限制：
- API: 100 请求/分钟/IP
- P2P: 50 消息/秒/IP

**Q: 私钥丢了怎么办？**
私钥=钱包所有权。如果丢失，Credits 和 Reputation 无法恢复。请妥善备份。

---

## 下一步

1. 准备节点机器（配置 config.yaml）
2. 等待 P2P 网络 beta 上线通知
3. 加入 Discord/Telegram 获取引导节点地址
4. 启动节点，开始赚取 Credits！
