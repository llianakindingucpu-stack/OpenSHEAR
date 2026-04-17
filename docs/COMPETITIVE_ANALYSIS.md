# DecentralAI 竞品与技术参考分析

> 最后更新：2026-04-16
> 目的：系统性记录所有竞品调研和技术参考项目，为后续设计决策提供对比依据

---

## 目录

1. [推理引擎对比](#1-推理引擎对比)
2. [分布式推理竞品全景](#2-分布式推理竞品全景)
3. [联邦学习与区块链相关](#3-联邦学习与区块链相关)
4. [非 Transformer 架构候选](#4-非-transformer-架构候选)
5. [关键借鉴点汇总](#5-关键借鉴点汇总)
6. [DecentralAI 差异化定位](#6-decentralai-差异化定位)

---

## 1. 推理引擎对比

### 1.1 vLLM（vllm-project/vllm）

**一句话**：工业级高吞吐 LLM 推理引擎，2000+ 贡献者，事实标准。

| 维度 | 详情 |
|------|------|
| 语言 | Python |
| 出身 | UC Berkeley Sky Computing Lab |
| 部署方式 | pip install + 依赖链 |
| 硬件需求 | 推荐 GPU（NVIDIA/AMD/Intel），支持 CPU |
| Star/Fork | 30K+（顶级项目） |

**核心能力：**
- **PagedAttention**：显存碎片管理，KV Cache 按页分配，8GB 显存跑更大模型
- **Continuous Batching**：多请求并行处理，动态批大小，高吞吐
- **Chunked Prefill + Prefix Caching**：预填充分块 + 前缀缓存，减少重复计算
- **Multi-LoRA**：原生支持多 LoRA 热切换，零停机加载新 adapter
- **MoE 推理**：原生支持 DeepSeek-V3/Qwen-MoE/GPT-OSS 等 MoE 模型
- **Speculative Decoding**：EAGLE/n-gram/suffix/DFlash，草稿-验证分离
- **量化**：FP8/MXFP8/MXFP4/NVFP4/INT8/INT4/GPTQ/AWQ/GGUF/compressed-tensors 等
- **并行模式**：Tensor/Pipeline/Data/Expert/Context 五种并行
- **Disaggregated Prefill/Decode**：解耦前后处理，分布式部署基础
- **OpenAI 兼容 API**：/v1/chat/completions 标准，含 Anthropic Messages API + gRPC
- **模型支持**：200+ HF 模型架构（LLM/MoE/SSM/多模态/Embedding/Reward）

**对 DecentralAI 的价值：**
- L2/L3/L4 节点的首选推理引擎
- Multi-LoRA = "专家委员会"的技术实现（base model + 多 LoRA adapter 热切换）
- Speculative Decoding = 五级节点调度理论基础（L1 草拟 → L2/L3 验证）
- Expert Parallelism = MoE 跨卡/跨节点推理的成熟方案
- Disaggregated Prefill/Decode = L1 做预填充 + L2 做解码的解耦调度

**局限：**
- 依赖 Python + PyTorch，部署门槛高
- 不适合 L0/L1 轻量节点
- 单项目独大，社区锁定风险

---

### 1.2 Shimmy（Michael-A-Kuykendall/shimmy）

**一句话**：极简主义推理服务器，Rust 单二进制，零配置部署。

| 维度 | 详情 |
|------|------|
| 语言 | Rust |
| 部署方式 | 单文件下载即用（shimmy.exe） |
| 硬件需求 | CPU 就能跑，自动检测 GPU |
| 许可 | 免费（MIT-like），"FREE forever" |
| 版本 | v1.9.0+ |

**核心能力：**
- **单二进制部署**：一个 exe 文件，不需要 Python/CUDA/conda
- **100% OpenAI API 兼容**：直接替换 OpenAI endpoint
- **GGUF + SafeTensors**：支持主流量化格式
- **模型热切换**：运行时切换模型，不重启服务
- **自动发现**：自动扫描 HuggingFace cache / Ollama / 本地目录
- **自动检测 LoRA adapter**：加载模型时自动发现关联 LoRA
- **CPU/GPU 混合 MoE 卸载**：70B+ 模型跑消费级硬件（--cpu-moe --n-cpu-moe 8）
- **多 GPU 后端**：CUDA（NVIDIA）/ Vulkan（跨平台）/ OpenCL（AMD/Intel）/ MLX（Apple Silicon）
- **智能后端降级**：指定 CUDA 但没有 NVIDIA GPU → 自动 fallback 到 Vulkan/OpenCL/CPU
- **Vision 支持**：MiniCPM-V 等多模态模型
- **零配置**：无 config file，无 setup wizard，端口自动分配避免冲突

**对 DecentralAI 的价值：**
- L0/L1 节点的理想推理引擎（CPU-first，单文件部署）
- 零配置 = 节点加入网络的最低门槛（下载 exe + 模型文件 → 运行 → 自动注册）
- OpenAI 兼容 API = DecentralAI Router 统一接口的参考实现
- CPU MoE 卸载 = 让无 GPU 节点也能跑 7B 模型
- 自动发现 = 节点上线自动注册到网络的模式参考

**局限：**
- 早期项目，个人维护，无企业背书
- 批处理/显存管理不如 vLLM 成熟
- Multi-LoRA 支持较基础（自动检测但不支持高级热切换）
- 无 Speculative Decoding

---

### 1.3 推理引擎分层部署策略

```
┌─────────────────────────────────────────────────────────┐
│                   DecentralAI Router                     │
│              （统一 OpenAI 兼容 API）                     │
└─────────┬──────────┬──────────┬──────────┬──────────────┘
          │          │          │          │
     ┌────▼───┐ ┌───▼────┐ ┌──▼────┐ ┌──▼──────┐
     │  L0    │ │  L1    │ │  L2   │ │ L3/L4   │
     │Shimmy  │ │Shimmy  │ │ vLLM  │ │  vLLM   │
     │CPU only│ │CPU/GPU │ │GPU    │ │ GPU 多卡│
     │验证/采集│ │轻量推理│ │标准推理│ │重度推理  │
     └────────┘ └────────┘ └───────┘ └─────────┘
```

**核心原则**：节点对外暴露统一的 OpenAI 兼容 API，Router 不关心内部用什么引擎。Shimmy 和 vLLM 不是竞争关系，而是互补的分层方案。

---

## 2. 分布式推理竞品全景

### 2.1 Cuckoo Network（cuckoo-network/cuckoo）

| 维度 | 详情 |
|------|------|
| Star/Fork | 408★ / 43F |
| 技术栈 | Go + Node.js/Yarn |
| 模式 | 区块链 + GPU 共享市场 |
| 核心思路 | 节点出租 GPU 算力，平台撮合请求方和提供方 |

**与 DecentralAI 的区别**：
- Cuckoo = GPU 算力租赁市场（类似去中心化的 AWS）
- DecentralAI = 分布式 MoE 专家委员会（异构节点协作推理）
- Cuckoo 的节点是"无脑的 GPU"，DecentralAI 的节点是"有智慧的专家"

---

### 2.2 BloomBee（ai-decentralized/BloomBee）

| 维度 | 详情 |
|------|------|
| 机构 | PASA Lab, UC Merced |
| 技术栈 | 基于 Hivemind + FlexLLMGen + Petals |
| 模式 | P2P 层分片推理 + 微调 |
| 支持模型 | LLaMA / BLOOM / Falcon / Mixtral |
| 特性 | Speculative decoding, micro batching, 无损压缩 |

**与 DecentralAI 的区别**：
- BloomBee = 层分片串行推理（把模型按层拆分到不同节点）
- DecentralAI = 独立专家并行推理（每个节点是完整的小模型+LoRA）
- 层分片有单点故障（任何一层挂了整条链断），专家模式天然容错

---

### 2.3 Petals（bigscience-workshop/petals）

| 维度 | 详情 |
|------|------|
| 模式 | BitTorrent 式层分片推理 |
| 性能 | Llama-2 70B 单批 ~6 tok/s |
| 特性 | DHT 节点发现，分布式 prompt-tuning |

**与 DecentralAI 的区别**：同 BloomBee，串行层依赖 vs Router 并行调度。

---

### 2.4 FedMoECap（ManosXen/...）

| 维度 | 详情 |
|------|------|
| Star/Fork | 1★ / 学术项目 |
| 核心创新 | Activation Freeze Criterion（按激活率冻结专家） |
| 策略 | Static / Pruning / Rolling 三种收敛策略 |
| 验证 | Jetson Orin + A100 双平台 |

**对 DecentralAI 的价值**：
- **Rolling 策略**与 DecentralAI 进化引擎天然契合
- Activation Freeze = 低资源节点只做推理，高资源节点承担训练
- 通信量降 95%，边缘能耗降 53%
- **最接近 DecentralAI 技术栈的开源参考**（虽然影响力极低）

---

### 2.5 Block-LoRA（PrasannaGundumogula/Block-LoRa）

| 维度 | 详情 |
|------|------|
| Star/Fork | 0★ / 学术原型 |
| 核心思路 | 区块链 + 联邦 LoRA 微调 |
| 链上 | CID + hash 存储 |
| 链下 | IPFS 存储数据 |
| 验证 | Proof-of-Validation 投毒检测 |
| 信任 | Trust Score（接受 +50 / 拒绝 -100） |
| 合约 | Hardhat 智能合约 |

**对 DecentralAI 的价值**：
- Trust Score 积分机制 → DecentralAI 信誉系统参考
- 链上存 hash + 链下存数据 → DecentralAI 合约层设计参考
- 投毒检测 → 结果验证层的参考实现

---

### 2.6 其他项目

| 项目 | Star | 定位 | 相关度 |
|------|------|------|--------|
| infinigence/FUSCO | 118★ | MoE 分布式数据 shuffle（all-to-all） | MoE 通信层参考 |
| parity-protocol | 51★ | 可验证分布式计算 | 可信计算参考 |
| exo-explore/exo | - | 本地 AI 运行框架 | 本地部署参考 |
| 50RC3/vAIn | - | P2P 联邦学习 + 符号推理 | 联邦学习参考 |
| PermLLM | 0★ | 去中心化安全推理 | 学术参考 |
| opengraviton/mju | - | Petals-style + activation 压缩 | 压缩传输参考 |

---

## 3. 联邦学习与区块链相关

### 3.1 联邦 MoE 微调

- **FedMoECap**（见 2.4）：Rolling 策略 → 进化引擎 Evolve 阶段
- **Whisper-MOE**（wisebreadloaf/Whisper-MOE）：3★，TTT MLP backbone + 动态专家选择，跨模态 MoE 参考

### 3.2 链上验证

- **Block-LoRA**（见 2.5）：Trust Score + 投毒检测
- **parity-protocol**：沙箱执行 + 模块化验证
- **piterodml 两个项目**：ZK 验证 + 框架蓝图（无实际代码）

### 3.3 空白领域（GitHub 0 结果）

以下方向 GitHub 完全无对标项目，确认 DecentralAI 差异化空间：
- 去中心化知识蒸馏
- 自进化 AI
- 去中心化验证激励（QoL + reward model）
- 解耦推理调度（异构节点编排）
- Speculative Decoding 边缘推理

---

## 4. 非 Transformer 架构候选

> 详细分析见 ARCHITECTURE_RESEARCH.md

| 架构 | 推理复杂度 | KV Cache | CPU 推理 | 社区活跃度 | 推荐度 |
|------|-----------|----------|---------|-----------|--------|
| **RWKV-7 "Goose"** | O(1) per token | 无 | ⭐⭐⭐ 最优 | Apache 2.0, 443+ 仓库 | ⭐⭐⭐ L1 首选 |
| **Mamba-2** | O(n) 线性 | 无 | ⭐⭐ 良好 | 官方 + 社区 | ⭐⭐⭐ 长期潜力 |
| **xLSTM** | O(n) 线性 | 无 | ⭐⭐ | NX-AI 官方, 146 仓库 | ⭐⭐ 研究阶段 |
| **RetNet** | O(1) 并行 | 无 | ⭐⭐ | 微软研究院 | ⭐⭐ |
| **Jamba** | O(n) 混合 | 部分 | ⭐ | AI21 Labs | ⭐ |
| **Zamba2** | O(n) 混合 | 无 | ⭐⭐ | Zyphra | ⭐⭐ |

**RWKV-7 关键特性**：
- Linear time, constant space, no KV cache, infinite ctx_len, free sentence embedding
- CPU 推理性能最优（无 KV Cache = 无显存压力）
- Apache 2.0 许可（完全自由）
- GitHub 443+ 相关仓库，社区驱动

**NVIDIA 验证**：2025.8 Jet-Nemotron 用 Mamba2/RWKV 替换 Transformer 做边缘推理。

**推荐策略**：多架构节点
- L1 节点：RWKV（CPU 最优）
- L2 节点：Qwen/Coder（GPU Transformer）
- L3 节点：混合（Transformer + SSM）
- L4 数据中心：大参数 Transformer

---

## 5. 关键借鉴点汇总

### 5.1 技术实现层面

| 来源 | 借鉴点 | DecentralAI 应用 |
|------|--------|-----------------|
| vLLM Multi-LoRA | 多 LoRA 热切换 | 专家委员会：base + 多 adapter 零停机切换 |
| vLLM Speculative Decoding | 草稿-验证分离 | L1 草拟 → L2/L3 验证 |
| vLLM Disaggregated Prefill/Decode | 解耦前后处理 | 分布式节点调度 |
| vLLM Expert Parallelism | MoE 跨卡推理 | MoE 跨网络推理基础 |
| vLLM PagedAttention | 显存碎片管理 | L2+ 节点显存优化 |
| Shimmy 零配置 | 单文件部署 | 节点加入网络的最低门槛 |
| Shimmy OpenAI API | 标准接口 | Router 统一调度接口 |
| Shimmy CPU MoE | CPU/GPU 混合 | L0/L1 节点跑 7B 模型 |
| FedMoECap Rolling | 滚动进化策略 | 进化引擎 Evolve 阶段 |
| FedMoECap Activation Freeze | 按激活率冻结 | 低资源节点只推理不训练 |
| Block-LoRA Trust Score | 信誉积分 | 信誉加权调度 |
| Block-LoRA PoV | 投毒检测 | 结果验证层 |
| RWKV-7 O(1) | 无 KV Cache | CPU 推理最优方案 |

### 5.2 系统设计层面

| 来源 | 借鉴点 | DecentralAI 应用 |
|------|--------|-----------------|
| 人类社会隐喻 | 自组织激励调度 | 价格信号 + 信誉 + 进化三层机制 |
| Cuckoo GPU 市场 | 算力交易模式 | Credit 奖励机制参考 |
| BloomBee/Petals | 分布式推理尝试 | 层分片 vs 专家并行的优劣对比 |
| Shimmy 自动发现 | 节点零配置上线 | 节点注册 + 心跳 + 自动发现 |

---

## 6. DecentralAI 差异化定位

### 6.1 竞品覆盖矩阵

```
                    推理引擎    分布式调度    经济激励    自进化    异构架构
vLLM                ██████░░░  ██░░░░░░░░  ░░░░░░░░░  ░░░░░░░░░  ░░░░░░░░░
Shimmy              ████░░░░░  █░░░░░░░░░  ░░░░░░░░░  ░░░░░░░░░  ███░░░░░░
Cuckoo              ░░░░░░░░░  ███░░░░░░░  █████░░░░  ░░░░░░░░░  ░░░░░░░░░
BloomBee            ███░░░░░░  █████░░░░░  ░░░░░░░░░  ░░░░░░░░░  ░░░░░░░░░
Petals              ███░░░░░░  █████░░░░░  ░░░░░░░░░  ░░░░░░░░░  ░░░░░░░░░
FedMoECap           ██░░░░░░░  ███░░░░░░░  ░░░░░░░░░  ██░░░░░░░  ░░░░░░░░░
Block-LoRA          ░░░░░░░░░  ██░░░░░░░░  ███░░░░░░  ░░░░░░░░░  ░░░░░░░░░
──────────────────────────────────────────────────────────────────────────
DecentralAI         ██████░░░  ██████░░░░  ██████░░░░  ██████░░░░  ██████░░░░
（目标）             用vLLM+Shimmy  dMoE+五级节点  Credit+信誉  Neuroplastic  RWKV+Qwen
```

### 6.2 独创交叉点（无竞品对标）

1. **dMoE + 异构架构**：不同节点跑不同架构（RWKV/Qwen/Mamba），Router 统一调度
2. **自进化闭环**：推理 → 收集失败 → 构造训练集 → LoRA 微调 → 验证 → 切换新 LoRA
3. **五级节点社会隐喻**：L0 搬砖到 L4 造火箭，人人有路，人人有回报
4. **信誉 + 进化双驱动**：信誉决定调度优先级，进化决定能力上限，两者互相促进
5. **从 CPU 到 GPU 的全谱系覆盖**：树莓派到数据中心，同一个网络

### 6.3 一句话定位

> DecentralAI 不是另一个推理引擎，也不是另一个 GPU 市场。
> 它是让任何设备都能成为 AI 专家的基础设施层——
> 通过经济激励和自进化机制，将异构节点的碎片算力组织成统一的智能网络。

---

*本文档随项目进展持续更新。新发现的项目和技术方向将追加到对应章节。*
