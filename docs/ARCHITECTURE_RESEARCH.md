# DecentralAI 架构研究报告 v2

> 日期：2026-04-16（更新） | 目的：探索非 Transformer 架构 + GitHub 竞品调研 + 可借鉴方案

---

## 第一部分：非 Transformer 推理方案（v1 保留）

### 一、为什么不能只看 Transformer

```
Transformer 的核心瓶颈：

  1. O(N²) 注意力复杂度
     → 序列越长，计算量和显存指数级增长
     → 8GB 显存跑 7B 已经勉强，长上下文直接爆

  2. KV Cache 显存爆炸
     → 每个 token 都要缓存 K/V 向量
     → batch size 大了，显存撑不住

  3. 训练和推理都需要大量 GPU 内存
     → 门槛高，只有有 GPU 的人能参与
     → 违背"让廉价设备也能贡献"的目标
```

**关键洞察**：去中心化网络的节点需要的是"推理够快 + 显存够省 + CPU 也能跑"，Transformer 在这三点上都不是最优解。

### 二、六大非 Transformer 架构候选

#### 1. Mamba / Mamba-2（状态空间模型 SSM）⭐⭐⭐⭐⭐

```
论文：Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)
作者：Albert Gu (CMU) + Tri Dao (Princeton)

核心原理：
  ├── 选择性状态空间模型（Selective SSM）
  ├── 递归更新隐藏状态（类似 RNN，但可并行训练）
  ├── 时间复杂度 O(N)（线性！不是二次方）
  └── 推理时不需要 KV Cache

关键优势：
  ├── 推理速度：比 Transformer 快 2-5x（同参数量）
  ├── 显存占用：比 Transformer 低 50-70%（无 KV Cache）
  ├── 长上下文：无限长度理论可行（线性增长）
  └── CPU 推理：比 Transformer 友好得多

开源模型：
  ├── Mamba-2（2024）：重构为结构化 SSM，训练更稳定
  ├── Mamba-2-2.7B：开源，性能接近同级别 Transformer
  └── Jamba（AI21）：Transformer + Mamba 混合，3x 吞吐量
```

#### 2. RWKV / RWKV-7 "Goose"（线性注意力 RNN）⭐⭐⭐⭐⭐

```
论文：Receptance Weighted Key Value (RWKV) (2023)
机构：RWKV 开源基金会（社区驱动，Apache 2.0）
最新版本：RWKV-7 "Goose"（2026，GitHub 活跃）

核心原理：
  ├── 线性注意力 + RNN 的混合体
  ├── 训练时像 Transformer（可并行），推理时像 RNN（O(1) 每步）
  ├── 完全不需要 KV Cache
  └── 时间复杂度 O(N) 训练，O(1) 推理

关键优势：
  ├── 推理显存极低（不需要存储历史 KV）
  ├── CPU 推理性能优秀（专为边缘设备优化）
  ├── 无限上下文窗口
  ├── free sentence embedding（RWKV-7 新增）
  └── 社区开源，非商业公司控制（符合去中心化理念）

GitHub：BlinkDL/RWKV-LM，443+ 相关仓库，活跃迭代中
```

#### 3. xLSTM（扩展长短期记忆）⭐⭐⭐⭐

```
论文：xLSTM: Extended Long Short-Term Memory (2024)
作者：Sepp Hochreiter（LSTM 原始作者！）

核心原理：
  ├── LSTM 的现代化改进
  ├── 引入指数门控（exponential gating）+ 矩阵记忆
  ├── 保留 RNN 的 O(N) 复杂度
  └── 可并行训练（解决了传统 LSTM 训练慢的问题）

关键优势：
  ├── 7B 模型已开源（xLSTM 7B，2025.1）
  ├── 推理速度快，显存低
  ├── 理论基础扎实（30 年 LSTM 研究积累）
  └── CPU 友好
```

#### 4. RetNet（保留网络）⭐⭐⭐

```
论文：Retentive Network: A Successor to Transformer for LLMs (2023)
机构：微软

核心原理：
  ├── 保留机制（Retention Mechanism）替代自注意力
  ├── 三种计算范式：并行/递归/因果注意力
  └── O(N) 复杂度，无 KV Cache

⚠️ 开源模型较少，社区不活跃
```

#### 5. Jamba（Transformer + Mamba 混合）⭐⭐⭐⭐

```
发布：AI21 Labs（2024.3）

核心原理：
  ├── 混合架构：Transformer 层（46层）+ Mamba 层（16层）
  ├── MoE（混合专家）：替代 MLP 层
  └── 推理时 3x Transformer 吞吐量

⚠️ AI21 Labs 商业公司，不符合完全开源理念
```

#### 6. Zamba2（Mamba + Transformer 混合）⭐⭐⭐

```
发布：Zyphra（2024）
├── Mamba 层 + 共享 Transformer 层的混合架构
├── 2.7B 参数
└── 速度比同参数 Transformer 快 2x，内存低 27%
```

### 三、架构对比矩阵

| 特性 | Transformer | Mamba-2 | RWKV-7 | xLSTM | RetNet | Jamba | Zamba2 |
|------|-------------|---------|--------|-------|--------|-------|--------|
| 训练复杂度 | O(N²) | O(N) | O(N) | O(N) | O(N) | O(N²)/O(N) | O(N) |
| 推理复杂度 | O(N)/步 | O(1)/步 | O(1)/步 | O(1)/步 | O(1)/步 | O(1)/步 | O(1)/步 |
| KV Cache | 需要 | 不需要 | 不需要 | 不需要 | 不需要 | 部分需要 | 不需要 |
| CPU 推理 | 差 | 中 | **优** | 优 | 优 | 中 | **优** |
| 显存占用 | 高 | 低 | **极低** | 低 | 低 | 中 | **极低** |
| 长上下文 | 贵 | 便宜 | **免费** | 便宜 | 便宜 | 便宜 | 便宜 |
| 生态成熟度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 开源理念 | 中立 | 学术 | **社区** | 学术 | 商业 | 商业 | 商业 |
| 最大开源模型 | 100B+ | 2.7B | **14B** | **7B** | - | 52B | 2.7B |
| LoRA 微调支持 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### 四、NVIDIA Jet-Nemotron（2025.8）——行业信号

```
NVIDIA 发布 Jet-Nemotron：混合架构语言模型
  ├── 将 Transformer attention 部分替换为 Mamba2、GLA、RWKV 等高效架构
  ├── 针对边缘设备优化
  └── 信号：NVIDIA 认为边缘推理 ≠ Transformer
```

---

## 第二部分：GitHub 竞品与参考项目调研（v2 新增）

> 调研方法：GitHub API search（topic + keyword），2026-04-15~16
> 搜索维度：decentralized LLM inference / distributed MoE / federated learning / verifiable inference / knowledge distillation / speculative decoding 等

### 一、赛道空白度确认

```
搜索 "decentralized LLM inference"  → 41 仓库
搜索 "distributed MoE inference"      → 6 仓库
搜索 "federated learning LLM decentralized" → 9 仓库
搜索 "verifiable inference ZK LLM"    → 0 仓库
搜索 "knowledge distillation decentralized LLM" → 0 仓库
搜索 "self-evolving AI model autonomous" → 0 仓库
搜索 "token incentive decentralized compute" → 0 仓库

结论：dMoE + 异构架构 + 自进化的交叉点，GitHub 上无对标项目。
```

### 二、高价值竞品/参考项目（按相关度排序）

#### 🏆 1. Cuckoo Network — 最成熟竞品

```
仓库：cuckoo-network/cuckoo
Stars：408 | Forks：43 | License：Apache 2.0
语言：TypeScript + Go | 创建：2024.04

定位：Decentralized AI Platform, GPU-sharing for text-to-image + LLM inference

架构：
  ├── Go 编写的节点客户端
  ├── TypeScript 前端（Web + Telegram/Discord Bot）
  ├── 节点配置通过 .env 文件
  └── 实际产品已上线（Telegram + Discord 可直接使用）

与 DecentralAI 的本质区别：
  ├── Cuckoo = GPU 租赁 marketplace（共享算力）
  ├── DecentralAI = 专家委员会（共享能力）
  └── 每个节点出租空闲 GPU vs 每个节点提供独立专家能力

借鉴价值：
  ├── ⭐⭐⭐ 产品化路径参考（如何从技术到实际产品）
  ├── ⭐⭐ 社区运营经验（Discord/TG 社区建设）
  └── ⭐ 经济模型设计（GPU 共享定价）
```

#### 🏆 2. BloomBee — Petals 改进版，技术最接近

```
仓库：ai-decentralized/BloomBee
Stars：中等 | Forks：11 | License：开源
机构：UC Merced PASA Lab + Yotta Labs

定位：Decentralized LLMs fine-tuning and inference with offloading

架构：
  ├── DHT（Distributed Hash Table）节点发现
  ├── libp2p P2P 通信
  ├── 按层分片（Layer 0-15 → Worker A，16-31 → Worker B...）
  ├── 客户端跑 embedding + LM head，Worker 跑中间层
  ├── 支持推理 + 微调
  └── 最低 4GB VRAM 即可参与（~4GB per worker）

关键特性（2025-2026 更新）：
  ├── Speculative Decoding + 裁剪（PR #38，2026.01）
  ├── Micro Batching + 无损压缩（PR #39，2026.02）
  ├── Weight Cache + Batch（PR #36，2025.11）
  ├── 支持模型：LLaMA 2/3、BLOOM、Falcon、Mixtral
  ├── HuggingFace 兼容（AutoDistributedModelForCausalLM）
  └── Colab 一键体验

上游依赖：
  ├── Hivemind（PyTorch 去中心化深度学习库）
  ├── FlexLLMGen（弱 GPU offloading 系统）
  └── Petals（BigScience 分布式 LLM）

与 DecentralAI 的区别：
  ├── BloomBee = 层分片（同一个模型拆开）
  ├── DecentralAI = 独立专家（每个节点独立模型 + LoRA）
  ├── BloomBee 严格串行 → DecentralAI Router 并行调度
  └── BloomBee 一节点挂 → 整条链断 → DecentralAI 专家冗余切换

借鉴价值：
  ├── ⭐⭐⭐ DHT + libp2p 节点发现方案
  ├── ⭐⭐⭐ Speculative Decoding 加速推理
  ├── ⭐⭐ Micro Batching + 无损压缩
  └── ⭐ HuggingFace 兼容层设计
```

#### 🏆 3. FedMoECap — MoE 联邦微调（最匹配 DecentralAI 需求）⭐⭐⭐

```
仓库：ManosXen/Routing-Aware-Federated-Fine-tuning-of-Mixture-of-Experts-LLMs
Stars：1 | 创建：2026.02 | 学术论文实现
平台：Jetson AGX Orin（边缘） + A100/DGX（服务器）

定位：FedMoECap - Resource-efficient Federated Learning for Mixture-of-Experts

四大核心机制：
  1. Activation Freeze Criterion
     ├── 全局池化所有层的所有专家
     ├── 按激活率选 top X% 专家训练
     ├── 允许整层冻结（资源不够时）
     └── 比传统按层设阈值更高效

  2. Selective LoRA
     ├── 只对选中（未冻结）的专家施加 LoRA
     ├── 通信量：~10MB per client（vs 完整模型 ~10GB）
     └── 通信量降低 95%

  3. 三种收敛策略（核心亮点）
     ├── Static（S）：首轮选定专家，训练中不变
     ├── Pruning（P）：逐步冻结已收敛专家，持续减少通信/能耗
     └── Rolling（R）：已收敛专家冻结 → 解冻新专家训练 → 轮换
     └── Rolling 在固定资源预算下最大化训练专家数量

  4. Hybrid Aggregation
     ├── 软权重保留已收敛客户端参数
     ├── 防止灾难性遗忘
     └── 特别适合 Rolling 策略

性能数据（OLMOE-1B-7B，PIQA/BoolQ/CSQA）：
  ├── 通信开销降低 95%
  ├── 边缘设备能耗降低 53%
  └── 精度保持竞争力

与 DecentralAI 的映射关系：
  ┌──────────────────────┬──────────────────────────────┐
  │ DecentralAI 需求      │ FedMoECap 方案              │
  ├──────────────────────┼──────────────────────────────┤
  │ 五级异构节点参与       │ Activation Freeze Criterion  │
  │                       │ 每节点独立 freeze rate       │
  ├──────────────────────┼──────────────────────────────┤
  │ LoRA 协作微调         │ Selective LoRA，10MB 级通信  │
  ├──────────────────────┼──────────────────────────────┤
  │ Neuroplasticity Engine│ Rolling 策略完美匹配         │
  │ 进化循环              │ Observe→Evolve→Verify 轮换   │
  ├──────────────────────┼──────────────────────────────┤
  │ 防止节点退化          │ Hybrid Aggregation 防遗忘    │
  └──────────────────────┴──────────────────────────────┘

借鉴价值：⭐⭐⭐（直接映射到 DecentralAI 三大核心机制）
```

#### 4. Block-LoRA — 区块链 + 联邦 LoRA 验证

```
仓库：PrasannaGundumogula/Block-LoRa
Stars：0 | 创建：2026.03 | 学术 Demo
语言：Python + Solidity（Hardhat）

定位：Blockchain-Enabled Federated Fine-Tuning of LLMs

架构流程：
  客户端 → 训练 LoRA → 上传 IPFS → 提交 CID 到链上
                                            ↓
                                      验证者下载评估
                                            ↓
                                      投票 接受/拒绝
                                            ↓
                                      智能合约决定
                                            ↓
                                      聚合器合并（只合并通过的）

关键机制：
  ├── Proof-of-Validation：验证者投票制
  ├── 三层投毒检测：准确率 <70% 拒绝 + 偏差 >50% 拒绝 + 后门测试
  ├── 信任积分：接受 +50，拒绝 -100，影响聚合权重
  ├── 智能合约接口：startRound / submitUpdate / submitVote / finalizeRound
  └── 只传 CID + hash 到链上，数据存 IPFS（链上存 hash，数据链下）

与 DecentralAI 的映射：
  ├── 链上存证方案 → 完全一致（DESIGN.md 已有设计）
  ├── 投毒检测 → P1 结果验证的具体实现参考
  ├── 信任积分 → DecentralAI 信誉加权系统模板
  └── 合约接口 → 合约设计的直接模板

注意：只是学术 Demo，但思路完整，架构图清晰

借鉴价值：⭐⭐⭐（合约层 + 验证层模板）
```

#### 5. Petals — BitTorrent 式分布式推理（鼻祖）

```
仓库：bigscience-workshop/petals
Stars：高 | 机构：BigScience（BLOOM 团队）
论文：arXiv 2209.01188

定位：Run large language models at home, BitTorrent-style

核心：
  ├── 按层分片，405B 模型分布式运行
  ├── 单批次推理：Llama-2 70B ~6 tok/s，Falcon 180B ~4 tok/s
  ├── 支持推理 + prompt-tuning
  ├── 公有 swarm（开放）+ 私有 swarm（信任组内）
  ├── Docker + WSL + macOS 全平台
  └── 已有在线聊天界面 chat.petals.dev

借鉴价值：
  ├── ⭐⭐ DHT 节点发现（与 BloomBee 一致）
  ├── ⭐⭐ 私有 swarm 的隐私分组思路
  └── ⭐ 公有网络运营经验
```

#### 6. FUSCO — MoE 分布式通信优化

```
仓库：infinigence/FUSCO
Stars：118 | Forks：11 | License：Apache 2.0
语言：Python | 创建：2025.12

定位：High-performance distributed data shuffling (all-to-all) for MoE training and inference

核心：
  ├── 解决 MoE 分片后跨节点 all-to-all 通信瓶颈
  ├── 纯技术组件，可独立使用
  └── Infinigence（国内公司）出品

借鉴价值：⭐⭐⭐（DecentralAI MoE 层间通信可直接复用）
```

#### 7. parity-protocol — 可验证分布式计算

```
仓库：theblitlabs/parity-protocol
Stars：51 | Forks：3 | 创建：2025.04

定位：Decentralized protocol for verifiable distributed compute
  ├── Sandboxed execution（沙盒执行）
  ├── Modular verification（模块化验证）
  └── LLM inference + federated learning at scale

借鉴价值：⭐⭐（验证层设计参考）
```

#### 8. 其他值得关注的项目

| 项目 | Stars | 亮点 | DecentralAI 关联 |
|------|-------|------|------------------|
| vAIn (50RC3/vAIn) | 中 | P2P + 协作学习 + 联邦学习 + 符号推理 + RL | P2P 协作学习思路 |
| mju (opengraviton/mju) | 新 | Petals 改进版，激活值压缩，降低通信开销 | 分片推理通信优化 |
| exo-explore/exo | 中高 | 消费级设备分布式推理 | 多设备调度架构参考 |
| cyber-inference | 11 | llama.cpp Web GUI，边缘部署 | 边缘部署 UI 参考 |
| cuckoo 竞品搜索 0 结果方向 | - | knowledge distillation + ZK inference + token incentive | 全部空白，DecentralAI 需自研 |

### 三、竞品全景总结

```
                算力共享          层分片          MoE 联邦          验证层
              ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  Cuckoo  │  │ BloomBee │  │FedMoECap │  │Block-LoRA│
              │  408⭐   │  │  Petals  │  │          │  │parity    │
              └──────────┘  └──────────┘  └──────────┘  └──────────┘
                    │              │              │              │
                    └──────────────┴──────────────┴──────────────┘
                                          │
                                   DecentralAI
                              dMoE + 异构 + 自进化
                            （交叉点，无直接竞品）
```

---

## 第三部分：三个最值得借鉴的思路（v2 新增）

### 思路 1：FedMoECap Rolling → DecentralAI 进化引擎

```
FedMoECap Rolling 策略本质：
  专家 A 收敛 → 冻结 A → 解冻专家 B → B 收敛 → 轮换...

映射到 DecentralAI 五级节点：
  L4 数据中心：全局聚合 + Hybrid Aggregation 防遗忘
  L3 重度推理：Rolling 管理多个 LoRA 专家
  L2 标准推理：Selective LoRA 训练选中专家
  L1 轻量推理：Activation Freeze 到极致（只推理不训练）
  L0 采集者：推理验证 + 数据收集

关键数据：通信量降低 95%，能耗降低 53%，精度保持
```

### 思路 2：Block-LoRA 信任积分 → DecentralAI 信誉系统

```
信任积分机制：
  推理质量好 → 信誉分上升 → 更多请求 → 更多数据 → 更好 LoRA → 正循环
  投毒/恶意 → 信誉分暴跌 → 请求减少 → 经济惩罚

参数参考（Block-LoRA）：
  ├── MIN_ACCURACY = 70%
  ├── MAX_DIVERGENCE = 50%
  ├── 接受 +50 信任分
  ├── 拒绝 -100 信任分
  └── 信任分影响聚合权重

这是自进化生态的基石：质量驱动分配，分配驱动进化。
```

### 思路 3：Speculative Decoding → 五级节点层次调度

```
推理加速方案：
  L0/L1 草拟（小模型 0.5B-1.5B，快速生成候选）
      ↓
  L2/L3 验证（大模型 7B+，并行验证多个候选）
      ↓
  接受/修正 → 返回结果

效果：
  ├── 小模型草拟几乎不花钱（CPU 就能跑）
  ├── 大模型只做验证（减少 forward pass）
  ├── 总体推理成本降低 2-3x
  └── 完美匹配五级节点的层次设计
```

---

## 第四部分：对 DecentralAI 的综合建议

### 多架构异构节点（保留 v1）

```
L0 采集者：CPU only，任何设备
L1 轻量推理：RWKV-7（CPU 最优）/ xLSTM 7B
L2 标准推理：Qwen2.5-Coder-7B（Transformer，生态好）
L3 重度推理：Transformer 7B-14B + Mamba 混合
L4 数据中心：大参数 Transformer 70B+
```

### 技术栈选型建议（v2 新增）

```
通信层：libp2p（BloomBee/Petals 验证）+ FUSCO（MoE all-to-all）
节点发现：DHT（成熟方案）
合约层：参考 Block-LoRA 架构（CID + hash 链上，数据 IPFS 链下）
进化引擎：FedMoECap Rolling 策略 + Selective LoRA
信誉系统：Block-LoRA Trust Score 模型
推理加速：Speculative Decoding（小节点草拟 + 大节点验证）
验证层：三层（请求方标注 + 冗余共识 + 信誉加权）+ Block-LoRA 投毒检测
```

### MVP 调整（保留 v1 + 补充）

```
Step 1a：Transformer 验证（Qwen2.5-Coder-7B，GPU 节点基准线）
Step 1b：SSM/RNN 验证（RWKV-7 7B，CPU 推理基准线）
Step 2：跨架构对比（同任务集，不同架构的质量/资源消耗）
Step 3：单节点进化闭环（FedMoECap Selective LoRA + Rolling）
Step 4：双节点协作（DHT 发现 + 层次推理）
Step 5：信誉系统原型（Block-LoRA Trust Score）
```

### 核心结论

```
1. 赛道空白确认：dMoE + 异构架构 + 自进化，GitHub 无直接竞品
2. FedMoECap 是技术基石：Rolling + Selective LoRA 直接映射进化引擎
3. Block-LoRA 是合约层模板：信任积分 + PoV + 智能合约
4. BloomBee/Petals 是网络层参考：DHT + libp2p + Speculative Decoding
5. Cuckoo 是产品化参考：GPU sharing marketplace 的运营经验
6. 多架构是正确路线：RWKV（CPU）+ Transformer（GPU）+ Mamba（混合）
7. 五级节点 + 层次调度是差异化核心：草拟-验证-兜底
```

---

## 附录：关键论文清单

| 论文 | 年份 | 机构 |
|------|------|------|
| Mamba: Linear-Time Sequence Modeling | 2023 | CMU + Princeton |
| Mamba-2 | 2024 | Tri Dao |
| RWKV: Receptance Weighted Key Value | 2023 | RWKV Foundation |
| RWKV-7 "Goose" | 2026 | RWKV Foundation |
| xLSTM: Extended Long Short-Term Memory | 2024 | JKU Linz |
| Retentive Network | 2023 | Microsoft |
| Jamba: Hybrid Transformer-Mamba | 2024 | AI21 Labs |
| Petals: Decentralized LLMs (arXiv 2209.01188) | 2022 | BigScience |
| FedMoECap: Routing-Aware Federated MoE | 2026 | 学术论文 |
| Block-LoRA: Blockchain Federated Fine-Tuning | 2026 | 学术论文 |
| Jet-Nemotron (NVIDIA) | 2025 | NVIDIA |

## 附录：关键 GitHub 仓库索引

| 仓库 | Stars | 关键词 |
|------|-------|--------|
| cuckoo-network/cuckoo | 408 | GPU sharing, decentralized AI |
| bigscience-workshop/petals | 高 | BitTorrent LLM, layer splitting |
| ai-decentralized/BloomBee | 中 | DHT, offloading, speculative decoding |
| infinigence/FUSCO | 118 | MoE all-to-all communication |
| BlinkDL/RWKV-LM | 高 | Linear attention, CPU inference |
| state-spaces/mamba | 高 | SSM, linear time |
| NX-AI/xlstm | 中 | Extended LSTM |
| theblitlabs/parity-protocol | 51 | Verifiable compute |
| ManosXen/Routing-Aware-Federated-MoE | 1 | Rolling strategy, Selective LoRA |
| PrasannaGundumogula/Block-LoRa | 0 | Blockchain, Trust Score |
| exo-explore/exo | 中高 | Consumer-grade distributed inference |
| 50RC3/vAIn | 中 | P2P AGI, collaborative learning |
| opengraviton/mju | 新 | Activation compression |
