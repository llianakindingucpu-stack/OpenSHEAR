# DecentralAI — 全链路推理跑通 (2026-04-17)

## 本次对话结论

**DecentralAI 推理全链路已跑通：**

```
Rust Router (:8082)  →  Python Worker (:8081)  →  RWKV-4-169M 推理  →  响应
```

## 做了什么

### 架构决策
- 放弃 candle-core：rand 版本冲突（candle 0.6 内部 rand-0.8，gemm 引入 half-2.7 → rand-0.9，无法共存）
- 推理走 Python（RWKV engine 已实现并验证），Rust 专注 Router + P2P 网络

### Rust Router 清理 / 编译
- 删除 `rwkv.rs`（不需要本地推理）
- 删除 `inference.rs` 引用
- 替换 `reqwest` → `ureq`（无 rand 依赖），修 API（`send_json`→`send_string`，`status().is_success()`→`200..300`）
- 修复 SQLite 路径（相对→绝对 `D:/IdeaProjects/decentral-ai/data/router.db`）
- 节点注册默认改为 Online

### 验证结果
```
Worker (:8081)    ✅  模型加载 1.6s，222 keys，vocab=50277
Router (:8082)    ✅  启动正常，注册/心跳/forward 端点就绪
端到端推理         ✅  6.2s，30 tokens，Router→Worker→RWKV 链路打通
```

## 当前文件结构
```
D:\IdeaProjects\decentral-ai\
├── tools/
│   ├── decentral-ai-core.exe  (5.44 MB, Node 主程序)
│   └── router.exe             (5.11 MB, Router 调度器)
├── src-rs/decentral-ai-core/  (Rust 项目源码)
│   ├── src/router.rs          (去中心化路由，Credits 结算，SQLite)
│   ├── src/network.rs          (P2P 网络，Gossip)
│   └── src/main.rs             (Node 主入口)
└── scripts/
    ├── inference_worker.py     (Python RWKV Worker，:8081)
    └── rwkv_engine.py          (RWKV-4-169M 推理引擎)
```

## 下一步（可并行推进）
1. **LoRA 微调**：RWKV-4-169M + unified_train.jsonl（已有数据集）
2. **Rust Router HTTP 端点**：/forward 完整对接 Worker，Credits 扣费
3. **P2P 节点互联**：两台机器运行 node，互发现 + Gossip
4. **Solidity 合约**：Credits 链上结算
