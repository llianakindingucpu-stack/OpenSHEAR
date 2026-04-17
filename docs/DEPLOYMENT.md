# DecentralAI 部署指南

> **前提条件**：Python 3.10+，网络连接正常

---

## 方式一：快速单节点（推荐入门）

### 1. 克隆项目

```bash
git clone https://github.com/your-username/decentral-ai.git
cd decentral-ai
```

### 2. 安装依赖

```bash
pip install websockets pyyaml pillow
```

**无 GPU 的机器**（CPU only）：
```bash
# RWKV-4-169M 只需 646MB RAM
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

**有 GPU 的机器**（NVIDIA）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm     # 高性能推理
```

### 3. 配置节点

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml:
#   node.level: L1          # L0/L1/CPU / L2/GPU / L3/Heavy / L4/DataCenter
#   node.model: rwkv-4-169m
```

### 4. 启动

```bash
# L0 采集者（CPU，¥2/月）
python run.py --level L0

# L1 轻量推理（0.5-1.5B，¥15/月）
python run.py --level L1

# 带 Web Dashboard（http://localhost:8000/）
python api_server.py --port 8000
```

---

## 方式二：API Server（OpenAI 兼容）

启动 API 网关，兼容所有 OpenAI SDK：

```bash
python api_server.py --port 8000
```

**SDK 调用示例：**

```python
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any"  # DecentralAI 无需 key

response = openai.ChatCompletion.create(
    model="rwkv-4-169m-pile",
    messages=[
        {"role": "system", "content": "你是一个专业的 Python 程序员。"},
        {"role": "user", "content": "写一个快速排序"}
    ],
    temperature=0.7,
    max_tokens=256
)
print(response['choices'][0]['message']['content'])
```

**cURL：**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-4-169m-pile",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 方式三：多节点 P2P 网络

至少 2 台机器，每台按"方式一"安装好 DecentralAI。

### 节点 A（Bootstrap）

```yaml
# config.yaml - 节点 A
node:
  id: node_alpha
  level: L2
  capabilities: [code, analysis]
  bootstrap: true

network:
  host: 0.0.0.0
  port: 8001
  seeds: []  # 无需种子节点（自己是第一个）
```

```bash
python run.py --config config.yaml
# 输出: P2P listening on ws://0.0.0.0:8001
#       Bootstrap node, no seed required
```

### 节点 B/C（加入网络）

```yaml
# config.yaml - 节点 B
node:
  id: node_beta
  level: L1
  capabilities: [analysis, general]
  bootstrap: false

network:
  host: 0.0.0.0
  port: 8002
  seeds:
    - ws://<节点A的IP>:8001   # 填写节点 A 的实际 IP
```

```bash
python run.py --config config.yaml
# 输出: Connected to 1 peers
#       Peer discovered: node_alpha
```

### 验证 P2P 连接

```bash
python src/ws_transport.py --demo
# 测试 ping/gossip/request 三种消息类型
```

---

## 方式四：Web Dashboard

实时可视化节点状态：

```bash
python api_server.py --port 8000
```

然后打开浏览器：
```
http://localhost:8000/
```

Dashboard 功能：
- 实时 peer 连接状态
- 推理请求统计（成功/失败率）
- Credit 余额变化
- Evolution 引擎状态
- 模拟网络拓扑动画

---

## 方式五：硬件完整验证（推荐硬件配置）

### 硬件要求

| 级别 | 模型 | 显存/RAM | 成本 |
|------|------|----------|------|
| L0 采集者 | CPU only | 4GB RAM | ¥0（闲散机器） |
| L1 轻量 | 0.5-1.5B | 4-8GB RAM | ¥15/月电费 |
| L2 标准 | 7B INT4 | 8GB VRAM | ¥65/月电费 |
| L3 重度 | 14B+ | 16-24GB VRAM | ¥200/月电费 |
| L4 数据中心 | 70B | A100 80GB | ¥2000/月电费 |

### 推荐配置（¥1000 以内搞定）

```
CPU:    Intel i5-12400  (¥800)
主板:   B660  (¥500)
内存:   32GB DDR4  (¥400)
显卡:   RTX 3060 8GB  (¥1000二手)
电源:   650W  (¥300)
机箱:   ¥200
SSD:    512GB  (¥200)

总计:  ~¥3400（一次性投入）
月电费: ~¥65（24h运行）
```

### 运行完整基准

```bash
# HumanEval 基准测试
python scripts/humaneval_baseline_full.py

# 结果会生成:
# results/humaneval_baseline_rwkv4_169m.json  (164题结果)
# results/humaneval_baseline_qwen2.5_7b.json   (对比基线)
```

### LoRA 微调流水线

```bash
# 收集失败题目 → 构造训练集 → LoRA 微调
python scripts/humaneval_lora_finetune.py \
    --baseline results/humaneval_baseline_rwkv4_169m.json \
    --output results/lora_model/

# 微调配置:
#   LoRA rank: 4/8/16/32
#   训练数据: 失败题目 + canonical_solution
#   硬件: RTX 3060 8GB 可训 rank=4
```

---

## 方式六：智能合约部署（可选）

### 前置

```bash
npm install -g hardhat
npx hardhat init
```

### 部署到本地 Hardhat

```bash
# 终端 1: 启动 Hardhat 节点
npx hardhat node
# 输出: HTTP & WebSocket provider on http://127.0.0.1:8545/

# 终端 2: 部署合约
npx hardhat run scripts/deploy.js --network localhost
```

### 部署到 Scroll（推荐）

```bash
# .env 配置
SCROLL_RPC=https://rpc.scroll.io
PRIVATE_KEY=0x...
DEPLOYER_ADDRESS=0x...

# 部署（手续费约 $5-10）
npx hardhat run scripts/deploy.js --network scroll
```

### 合约地址注册

部署完成后，将合约地址写入 config.yaml：

```yaml
contracts:
  credits:     "0x..."   # DecentralAICredits
  reputation:  "0x..."   # DecentralAIReputation
  settlement:  "0x..."   # DecentralAISettlement
  governance:  "0x..."   # DecentralAIGovernance
  rpc_url:     "https://rpc.scroll.io"
  private_key: "0x..."   # 节点运营私钥
```

---

## 环境变量速查

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DECENTRALAI_LEVEL` | L1 | 节点级别 |
| `DECENTRALAI_PORT` | 8001 | P2P 端口 |
| `DECENTRALAI_API_PORT` | 8000 | API 端口 |
| `DECENTRALAI_MODEL` | rwkv-4-169m | 默认模型 |
| `PYTHONPATH` | D:\pylib | 自定义库路径 |

---

## 常见问题

**Q: WebSocket 连接失败？**
```bash
# 检查防火墙
netsh advfirewall firewall add rule name="DecentralAI P2P" ^
  dir=in action=allow protocol=TCP localport=8001
```

**Q: 模型加载 OOM？**
```yaml
# config.yaml 降低精度
model:
  precision: fp16    # 默认 fp32
  # 或用 INT4 量化
```

**Q: pip install 被墙？**
```bash
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: HuggingFace 下载慢？**
```python
# 使用镜像
import huggingface_hub
huggingface_hub.login()
# 或设置 HF_ENDPOINT=https://hf-mirror.com
```

---

## 快速验证清单

```bash
# 1. 克隆 + 安装
git clone ... && pip install websockets pyyaml pillow

# 2. 启动节点（默认 L1）
python run.py --level L1

# 3. 打开 Dashboard（另一个终端）
python api_server.py

# 4. 浏览器打开
http://localhost:8000/

# 5. 跑测试
py -3 tests/test_core.py
# 期望: 33/33 OK

# 6. 完整 HumanEval 基线（需要 GPU，预算 ¥1000+）
python scripts/humaneval_baseline_full.py
```
