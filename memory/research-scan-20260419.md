# Research Scan — 2026-04-19

## ArXiv cs.NE (本周 4/13-4/17, 52 papers)

### 🔥 高度相关

**1. Structure as Computation: Developmental Generation of Minimal Neural Circuits**
- arXiv: 2604.15143 (2026-04-16)
- 模拟皮层神经发生发育过程，从单个干细胞出发，基于小鼠单细胞转录组数据的基因调控规则
- 5000 个细胞发育后仅 85 个成熟神经元（1.7%），但形成 200,400 个突触（平均 4715 度/神经元）
- 零训练 MNIST 90%+ 准确率，1 epoch 后；CIFAR-10 40.53%
- **对 SHEAR 启发**: "发育规则雕刻出领域通用的拓扑结构" — SHEAR 的 Cell 生成机制可以借鉴这种"生成即结构"思路。不是训练出连接，而是发育规则决定拓扑。极少数节点高连接 = Aggregator 的设计有生物学先例

**2. Beyond LLMs: VaCoAl — Hyper-Dimensional Computing for Reversible Multi-Hop Reasoning**
- arXiv: 2604.11665 (2026-04-12, v3)
- 基于 Galois 域代数的高维计算架构，发现确定性 STDP（spike-timing-dependent plasticity）涌现
- 稀疏分布式记忆(SDM) + 超高维二值空间 + Galois 域扩散
- 可逆组合推理、组合泛化、透明可靠性度量(CR score)
- 470K 知识图谱上 57 代追溯（25.5M 路径）
- **对 SHEAR 启发**: 解决灾难性遗忘、绑定问题。SDM 的分布式表示理念与我们 dMoE 的信誉系统互补。STDP 的涌现 = 路径依赖的语义选择，可借鉴为 Router 的专家选择机制

**3. The Dragon Hatchling: The Missing Link between Transformer and Brain Models**
- arXiv (2025-09), Kosowski et al.
- 统一、无尺度生物网络（如大脑）与 Transformer 的数学联系
- **对 SHEAR 启发**: 值得深读，可能为"为什么 RWKV（无尺度循环网络）比 Transformer 更像大脑"提供理论支撑

**4. Nonrandom, Non-Lamarckian Mutation in Evolution (IBE Theory)**
- arXiv: 2604.12884 (2026-04-15)
- Interaction-based Evolution: 突变非随机非拉马克，而是基因组内部信息跨代积累影响
- EDA 框架模拟：简约性(parsimony)与适应度(fitness)交互驱动进化
- **对 SHEAR 启发**: SHEAR 的节点进化机制可以借鉴 IBE — 不是纯随机变异，而是"内部信息积累 → 定向变异"。LoRA 参数的进化可以用类似方式：成功模式在基因组内积累，变异受历史约束

### ⚡ 中等相关

**5. DNN-guided PSO for Dynamic Environment Tracking**
- arXiv: 2604.14064, ISMSI 2026
- 中心化网络 + 分布式网络两种粒子群变体
- **对 SHEAR**: Router 的动态负载均衡可参考这种"预测移动最优位置"的思路

**6. When MoE Meets Blockchain (B-MoE) — 已撤稿**
- arXiv: 2509.12141 (2025-09, withdrawn)
- 区块链 + MoE 分布式框架，边缘层+区块链层+存储层
- 区块链追踪验证专家计算结果
- **对 SHEAR**: 虽然撤稿，但框架思路与 DecentralAI 一致。我们的 TOPLOC + 信誉系统是更轻量的替代方案，不需要区块链也能实现可信计算

## GitHub Trending (本周)

### 🔥 高度相关

**7. NousResearch/hermes-agent** ⭐ 99K (+47K 本周)
- 自改进 AI Agent，内置学习循环
- 从经验创建技能、使用中改进、跨会话知识持久化
- FTS5 会话搜索 + LLM 摘要 + 用户建模 (Honcho)
- Cron 调度器 + 多平台 (Telegram/Discord/Slack/WhatsApp/Signal)
- **对 SHEAR 启发**: hermes-agent 的"技能创建→改进→搜索"闭环与 EvoAgent 的 Phase 3 (Neuroplasticity) 高度一致。可直接参考其 skill schema 和 memory 架构

**8. EvoMap/evolver** ⭐ 4.9K (+2K 本周)
- GEP (Gene Expression Programming) 驱动的 Agent 自进化引擎
- 提示词调整 → 可审计、可复用的进化资产
- EvoMap 网络：Agent 通过验证协作进化
- 注意：作者声称 Hermes Agent 抄袭了其记忆/技能/进化设计
- **对 SHEAR 启发**: GEP 的"基因→表达式→适应度→选择"循环是 EvoAgent 进化引擎的理论基础。但 GEP 偏符号回归，SHEAR 需要的是参数级进化(LoRA)

**9. lsdefine/GenericAgent** ⭐ 4.2K (+2.4K 本周)
- 3.3K 行代码的极简自进化 Agent
- 9 个原子工具 + 100 行 Agent Loop
- 自动将执行路径结晶为技能
- 全仓库由 GenericAgent 自主创建（作者从未打开终端）
- **对 SHEAR 启发**: "极简种子 + 自举"理念验证。3K 行能实现完全自主 — 说明我们的 EvoAgent 冷启动策略可行

### ⚡ 间接相关

**10. shiyu-coder/Kronos** ⭐ 19K (+6.5K 本周) — 金融市场语言基础模型
**11. OpenBMB/VoxCPM2** ⭐ 14K (+5.8K 本周) — Tokenizer-free TTS 多语言语音生成
**12. coleam00/Archon** ⭐ 19K (+3.7K 本周) — AI 编码的确定性 harness builder

## 综合分析与对 SHEAR 的建议

### 新发现的关键启发

1. **极小网络高效学习有生物学先例** (2604.15143)
   - 85 个神经元 → 200K 突触 → 零训练 90% MNIST
   - SHEAR 不需要"大"模型，需要"对"的拓扑结构
   - **行动**: Cell 的连接模式应该模拟神经发生规则，而非随机初始化

2. **STDP 在代数系统中涌现** (2604.11665)
   - 不需要模拟脉冲，纯代数系统就能产生类 STDP 行为
   - SHEAR 的 Router 可以用类似机制：路径使用频率越高，连接越强
   - **行动**: 信誉系统的权重更新引入路径依赖衰减（类似 STDP 的时序窗口）

3. **IBE 进化理论** (2604.12884)
   - 突变非随机，受基因组内部信息积累影响
   - SHEAR 的 LoRA 进化应该是"约束变异"而非纯随机
   - **行动**: 保留成功 LoRA 的统计特征，变异在新参数空间内采样

4. **Agent 自进化已成主流赛道**
   - hermes-agent (99K), GenericAgent (4.2K), evolver (4.9K)
   - 但都在应用层，没有人在模型层做分布式进化
   - **SHEAR 的差异化**: 不是 Agent 层的技能积累，而是模型层的参数进化

### 赛道竞争格局更新

| 维度 | SHEAR | 竞品 |
|------|-------|------|
| 分布式推理 | ✅ 异构五级节点 | Petals (同构), B-MoE (已撤) |
| 模型层进化 | ✅ LoRA + dMoE | 无 |
| Agent 自进化 | ❌ 未涉及 | hermes-agent, GenericAgent |
| 脑科学启发的拓扑 | ✅ Cell+Aggregator | 需深化 |

### 推荐阅读优先级
1. ⭐⭐⭐ 2604.15143 — 发育规则生成最小神经回路（最直接启发）
2. ⭐⭐⭐ 2604.11665 — VaCoAl / STDP 涌现（Router 机制设计）
3. ⭐⭐ Dragon Hatchling — Transformer 与大脑的数学联系
4. ⭐⭐ 2604.12884 — IBE 进化理论（节点进化机制）
5. ⭐ GenericAgent 源码 — 极简自举验证
