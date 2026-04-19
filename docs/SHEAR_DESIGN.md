# Project SHEAR — Stateless Hybrid Ensemble Architecture for Reasoning

## 一、项目代号

**SHEAR** — 并行推理引擎
**全称**: Stateless Hybrid Ensemble Architecture for Reasoning

## 二、核心承诺

- 完全抛弃串行 Transformer / RNN 流水线
- 天生支持抢占式、多廉价 CPU、分布式并行
- 通信量极小，百兆网即可跑
- 目标：≥20 token/s，能力对标 RWKV-14B
- 全部用 Rust 实现
- 个人 / 小团队可训练、可部署

## 三、核心设计原则（铁律，绝不违反）

1. **无层级依赖**：Cell 之间无串行依赖
2. **无全局状态**：不依赖 KV Cache 同步，Cell 状态是局部的
3. **全可分片**：任意切分子任务，拔掉任意 Cell 照常运行
4. **抢占安全**：任务可重复下发，谁快用谁
5. **极小通信**：单次 Cell 结果 ≤ 512B

## 四、模型结构

```
[输入 Token Embedding]
      ↓ 广播
┌───────┬───────┬───────┬───────┐
Cell 0  Cell 1  Cell 2  ... Cell N  ← 全部并行、无依赖、无通信
│local  │local  │local  │       local│
│state  │state  │state  │       state│
└───┬───┴───┬───┴───┬───┴───────┬───┘
    │       │       │           │
    └───────┴───────┼───────────┘
                    ▼
           [Aggregator]
    学习到的路由权重（非固定均值）
                    ▼
           [Output Head]
        → next token probability
```

## 五、Cell 实现

### 当前实现（v0.2.0）

每个 Cell 基于 RWKV-4 架构，独立运行一个小型语言模型：

```rust
pub struct CellConfig {
    pub vocab_size: usize,    // 50277 (standard)
    pub d_model: usize,       // 768
    pub d_ffn: usize,         // 3072
    pub n_layers: usize,      // 6
    pub head_size: usize,     // 64
    pub n_heads: usize,       // 12
    pub max_seq_len: usize,   // 2048
}
// Total: ~200M params per Cell
```

### Cell 内部前向传播

```
Input Token
    ↓ Embedding [vocab_size, d_model]
    ↓ For each layer:
    │   ↓ LayerNorm (ln1)
    │   ↓ TimeMix (RWKV-style linear recurrence)
    │   │   wkv = (decay * state + key * value) / (decay + key)
    │   │   state_new = decay * state + key * value  ← O(1), 固定大小
    │   ↓ Output projection [d_model, d_model]
    │   ↓ Residual connection
    │   ↓ LayerNorm (ln2)
    │   ↓ FFN (SwiGLU: gate * silu(up))
    │   ↓ Residual connection
    ↓ LayerNorm (ln_out)
    ↓ Output Head [d_model, vocab_size]
    → logits
```

### 局部状态（CellState）

```rust
pub struct TimeMixState {
    pub aa: Vec<f32>,  // attention accumulator [n_heads]
    pub bb: Vec<f32>,  // attention accumulator [n_heads]
    pub pp: Vec<f32>,  // previous decay product [n_heads]
}
```

- 固定大小，不随序列长度增长
- 不需要跨 Cell 同步
- 不需要 KV Cache

## 六、Aggregator 实现

### 当前策略

| 策略 | 实现 | 适用场景 |
|------|------|---------|
| **WeightedSum** | `Σ(w_i · logits_i)` | 默认，适合多样化 Cell |
| **BestOfN** | 选 confidence 最高的 Cell | Cell 有明确专精时 |
| **RankBased** | 按 confidence 排名加权 | 折中方案 |
| **Adaptive** | 动态选择策略 | 最高质量，略多计算 |

### Confidence 计算

```
confidence_i = max(softmax(logits_i))
```

confidence 高 = Cell 对自己的输出很确定 → Aggregator 给更高权重。

## 七、Speculative Decoding（推测解码）

SHEAR 的 Cell 架构天然支持推测解码：

### Phase 1（已实现 ✅）

- N 个 Cell 共享模型权重（Arc<RwkvModel>，只读）
- 每个 Cell 独立状态 + 不同采样温度
- 逐 token 投票，rayon 并行
- 结果：1.1x 加速（并行消除了开销）

### Phase 2（设计完成 📐）

- L0/L1 节点做 draft（小模型，快）
- L2/L3 节点做 verify（大模型，准）
- Draft k tokens → Verify batch → Accept/Reject + rollback
- 预期加速：1.5-2x

详见 [Speculative Decoding 设计文档](SPECULATIVE_DECODING.md)

## 八、五级节点体系

| 级别 | 角色 | 硬件 | 模型 | 职能 |
|------|------|------|------|------|
| L0 | 采集者 | CPU only | 无 | 数据采集、路由转发 |
| L1 | 轻量推理 | CPU+4GB | 0.5B-1.5B | Draft 生成、轻量推理 |
| L2 | 标准推理 | 3060 | 7B | 标准推理、验证 |
| L3 | 重度推理 | 3090/4090 | 14B+ | 重度推理、LoRA 微调 |
| L4 | 数据中心 | A100/H100 | 70B+ | 全量训练、数据策展 |

详见 [节点体系文档](NODE_HIERARCHY.md)

## 九、为什么 SHEAR 能强？

### 集成学习理论

Mixture of Experts (MoE) 已在大规模验证：
- DeepSeek V3: 256 Experts, 671B → GPT-4 级别
- Mixtral 8x7B: 超越 LLaMA-2 70B

SHEAR = **分布式 MoE**，Expert 分布在不同机器上。

### 竞争选拔 vs 投票均值

```
传统集成：output = mean(A, B, C, ..., N)  → 趋于平庸
SHEAR：  output = best(A, B, C, ..., N)   → 保留专长
```

### 大脑皮层柱类比

大脑皮层 ~100 亿个柱状结构，每个柱独立处理信息，少量横向连接协调。大量弱单元并行 = 强智能。

## 十、参数规模与性能目标

| 配置 | Cell 数 | 总参数 | 单机需求 | 预期能力 |
|------|---------|--------|---------|---------|
| 最小 | 8 | 1.6B | 2GB × 8 | GPT-2 级别 |
| 标准 | 32 | 6.4B | 2GB × 32 | GPT-3 级别 |
| 大型 | 64 | 12.8B | 2GB × 64 | GPT-3.5 级别 |
| 全网 | 128+ | 25.6B+ | 分布式 | RWKV-14B 级别 |

**单机只需跑 1 个 Cell（200M, 2GB RAM）**

## 十一、当前基准测试

| 指标 | Baseline (1 Cell) | Ensemble (3 Cells, rayon) |
|------|-------------------|---------------------------|
| 速度 (CPU, 169M) | ~4.87 tok/s | ~2.4 tok/s (1.1x) |
| 共识率 | N/A | 45.6% |
| HumanEval Pass@1 | 0.0% (未训练) | TBD |
| 崩溃数 | 0/164 | 0 |

测试环境：Intel Pentium G4560, 8GB RAM, 无 GPU

详见 [基准测试文档](BENCHMARKS.md)

## 十二、技术栈

| 组件 | 技术 | 原因 |
|------|------|------|
| 推理引擎 | Rust (custom RWKV-4) | 性能、内存安全、无 GC |
| 并行 | rayon | 零开销数据并行 |
| HTTP API | axum | 异步、类型安全 |
| P2P | tokio | 全双工、低延迟 |
| Tokenizer | Custom BPE | 无 Python 依赖 |
| 数据库 | SQLite (rusqlite) | 嵌入式、零配置 |
| 权重格式 | 自定义 binary | 直接 mmap，无运行时开销 |
