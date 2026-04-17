# Project SHEAR — Stateless Hybrid Ensemble Architecture for Reasoning

## 一、项目代号

**SHEAR** — 并行推理引擎
**全称**: Stateless Hybrid Ensemble Architecture（无状态·混合集成·并行大模型）

## 二、核心承诺

- 完全抛弃串行 Transformer / RNN 流水线
- 天生支持抢占式、多廉价 CPU、分布式并行
- 通信量极小，百兆网即可跑
- 目标：≥20 token/s，能力对标 RWKV-14B
- 全部用 Rust 实现
- 个人 / 小团队可训练、可部署

## 三、核心设计原则（铁律，绝不违反）

1. **无层级依赖**：不搞 layer1→layer2→layer3 串行
2. **无全局状态**：不依赖持续隐状态、不依赖 KV Cache（Cell 自身状态是局部的，不需要跨 Cell 同步）
3. **全可分片**：任意切分子任务，无内部耦合；拔掉任意 Cell，剩余照常运行
4. **抢占安全**：任务可重复下发，谁快用谁
5. **极小通信**：单次分片结果 ≤ 512B

## 四、模型结构

```
[输入 Token Embedding]
      ↓ 广播
┌───────┬───────┬───────┬───────┐
Cell A  Cell B  Cell C  ... Cell N  ← 全部并行、无依赖、无通信
│local  │local  │local  │       local│
│state  │state  │state  │       state│
└───┬───┴───┬───┴───┬───┴───────┬───┘
    │       │       │           │
    └───────┴───────┼───────────┘
                    ▼
           [Aggregator]
    学习到的路由权重（非固定均值）
    不同 Cell 贡献度不同
                    ▼
           [Output Head]
        → next token probability
```

## 五、Cell 设计

每个 Cell 是一个独立的小型神经网络：
- **参数量**：~200M 每个
- **结构**：Embedding → 线性递归层（RWKV 式 time-mix）→ SwiGLU 前馈层 → Output
- **局部状态**：每个 Cell 维护自己的 time-mix 状态（线性递归），不需要 KV Cache，不需要跨 Cell 同步
- **无循环依赖**：Cell A 的输出不作为 Cell B 的输入
- **可选专精**：不同 Cell 可在不同领域数据上训练（代码/数学/语言/逻辑）
- **硬件需求**：单个 Cell 用 2GB 内存机器即可运行

### Cell 内部结构

```rust
struct Cell {
    embedding: Embedding,       // [vocab_size, d_model]
    time_mix: LinearRNN,        // RWKV-style time-mix, O(1) per token
    ffn: SwiGLU,                // Feed-forward with gated activation
    output: Linear,             // [d_model, vocab_size]
    state: Option<CellState>,   // Local recurrent state, NOT shared
}

impl Cell {
    fn forward(&mut self, token: TokenId) -> Tensor {
        let x = self.embedding.forward(token);
        let x = self.time_mix.forward(x, &mut self.state);  // local state update
        let x = self.ffn.forward(x);
        self.output.forward(x)
    }
}
```

### 线性递归层（time-mix）

区别于传统 RNN 的门控（LSTM/GRU），time-mix 是纯线性操作：

```
y = (w * x + k * state) / (w + k)

where:
  w = learned time-decay weights
  k = learned key projection
  state = accumulated from previous tokens (local to this Cell)
```

- O(1) per token（不随序列长度增长）
- 无 KV Cache（状态是一个固定大小的向量）
- 纯线性，可并行推理时无需等待前一步完成
- 每个时间步只需 ~3 次矩阵乘法

## 六、Aggregator 设计

Aggregator 负责将 N 个 Cell 的输出融合为最终 token 概率：

```
Input: [logits_A, logits_B, ..., logits_N]  (N 个 Cell 的输出)
       [confidence_A, confidence_B, ...]      (学习到的权重)

Output: weighted_logits = Σ (w_i * logits_i)
        next_token = argmax(weighted_logits)
```

### 路由策略（非固定均值）

- **学习到的权重**：Aggregator 根据输入内容动态调整每个 Cell 的权重
- **专精优先**：如果 Cell A 专精代码，输入是代码请求，Cell A 权重自动升高
- **抢占式**：如果某个 Cell 崩溃或超时，自动降低其权重，不影响其他 Cell
- **可扩展**：新增 Cell 不需要重新训练现有 Cell，只需更新 Aggregator 权重

## 七、为什么 SHEAR 能强？

### 7.1 集成学习理论

Mixture of Experts (MoE) 已在大规模验证：
- DeepSeek V3: 256 Experts, 671B total params → GPT-4 级别
- GPT-4 内部使用 MoE（非官方确认）
- Mixtral 8x7B: 8 Experts → 超越 LLaMA-2 70B

SHEAR 本质上就是 **分布式 MoE**，区别在于 Expert 分布在不同机器上。

### 7.2 信息融合 vs 信息瓶颈

| 方案 | Cell 间交互 | 信息瓶颈风险 | 可行性 |
|------|-----------|-------------|--------|
| 纯投票（原始设计） | 零 | 高 | 低 |
| 固定均值聚合 | 零 | 中高 | 低 |
| 学习到的路由权重 | 零（仅 Aggregator 读） | 中 | 中 |
| Cell 内线性递归 | 零（局部状态） | 低 | **高** |
| 专精 + 路由 | 零（Aggregator 学习路由） | 低 | **高** |

关键改进：通过 Cell 内的局部 time-mix，每个 Cell 独立理解上下文，不依赖跨 Cell 通信。32 个有各自"记忆"的 Cell 竞争输出，Aggregator 选最优。

### 7.3 竞争选拔 vs 投票均值

不是让所有 Cell 投票取平均（这会拉平差异），而是让所有 Cell **竞争**，Aggregator 选最强的输出：

```
传统集成：output = mean(A, B, C, ..., N)  → 趋于平庸
SHEAR：  output = best(A, B, C, ..., N)   → 保留专长
```

### 7.4 大脑皮层柱类比

大脑皮层由 ~100 亿个柱状结构组成，每个柱是一个独立的信息处理单元。柱之间有少量横向连接（类似 Aggregator），但每个柱能独立完成基本的感知和模式识别。

大量弱单元并行 = 强智能。这不是比喻，这是大脑的实际运作方式。

## 八、参数规模与性能目标

| 配置 | Cell 数 | 总参数 | 单机需求 | 预期能力 |
|------|---------|--------|---------|---------|
| 最小 | 8 | 1.6B | 2GB RAM × 8 | GPT-2 级别 |
| 标准 | 32 | 6.4B | 2GB RAM × 32 | GPT-3 级别 |
| 大型 | 64 | 12.8B | 2GB RAM × 64 | GPT-3.5 级别 |
| 全网 | 128+ | 25.6B+ | 分布式 | RWKV-14B 级别 |

**注意**：单机只需要跑 1 个 Cell（200M, 2GB RAM）。总参数量由网络规模决定。

## 九、训练策略

### 9.1 Cell 训练（独立、可并行）

每个 Cell 独立训练，互不影响：
1. 基础训练：通用语言建模数据（The Pile, RedPajama）
2. 专精训练：领域数据（代码/数学/语言/逻辑）
3. 硬件需求：单 GPU (RTX 3060 8GB) 或纯 CPU（慢但可行）

### 9.2 Aggregator 训练（轻量）

Aggregator 只需要学习路由权重：
1. 输入：N 个 Cell 的输出 logits
2. 标签：Ground truth next token
3. 损失：加权的交叉熵
4. 训练极快：参数量远小于 Cell

### 9.3 部署流程

```
训练完成 →
  ├── Cell A weights → 部署到 Node A
  ├── Cell B weights → 部署到 Node B
  ├── ...
  └── Aggregator weights → 部署到 Router Node
```

每个 Cell 可以独立更新（LoRA 微调 → 新快照），不影响其他 Cell。

## 十、与 DecentralAI 的关系

SHEAR 是 DecentralAI 项目的核心推理引擎重新设计。DecentralAI 保留了五级节点体系和智能合约层，SHEAR 替换了原有的"分布式 Transformer"思路，改为真正的并行 Cell 架构。

保留的 DecentralAI 组件：
- P2P 网络层（network.rs）
- Router 和负载均衡（router.rs）
- Credits + Reputation 系统
- Solidity 智能合约
- 节点自进化循环

替换的组件：
- ~~分布式 Transformer 分片~~ → SHEAR Cell 引擎
- ~~KV Cache 同步~~ → 无全局状态
- ~~层间通信~~ → 无层级依赖
