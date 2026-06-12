# 2026-04-17 DecentralAI Router 实现

## 目标
用 Rust 实现多节点推理调度器，替代 Python 方案（避免依赖地狱）

## 完成内容

### 1. Rust Router (`src-rs/decentral-ai-core/src/router.rs`)
- **NodeRegistry**：节点注册、心跳、后台健康检查（15 秒轮询）
- **RouterState**：线程安全（Arc+RwLock），含 HTTP client（reqwest）
- **智能负载均衡**：综合评分 = 成功率×40% + 速度×30% + 延迟×20% + 级别×10%
- **Credits 结算**：SQLite（rusqlite），预授权→实际消耗→差额退款
- **端点**：
  - `GET /health` — 健康检查
  - `GET /nodes` — 列出所有节点
  - `POST /register` — 注册节点
  - `POST /heartbeat` — 节点心跳（Offline→Online）
  - `DELETE /unregister/:node_id` — 注销节点
  - `GET /stats` — 路由统计
  - `GET /balance?user_id=` — 查询余额
  - `POST /forward` — 路由推理请求

### 2. 独立二进制入口 (`src/router_main.rs`)
- `cargo run --bin router -- --port 8082 --db-path ../data/router.db`
- 5.4MB release 二进制，无外部依赖

### 3. 模块重构
- 新增 `lib.rs` 导出共享类型（AppState、NodeRole、ChatMessage）
- `main.rs` 改为从 lib 导入，消除了重复定义
- `router.rs` 使用本地 ChatMessage（lib 内不能引用自身 crate 名）

### 4. 编译错误修复
- `axum::Extract` → `axum::extract::State`（axum 0.7 API）
- `query_row` 需要 3 参数（SQL, params, closure）
- `decentral_ai_core::ChatMessage` → 本地定义
- `AppState` 从 main.rs 移到 lib.rs 共享
- `pending_requests` dead_code warning（不影响）

## 端到端测试结果
```
POST /forward → RWKV Worker → Model
tokens: 30, latency_ms: 4629, credits_cost: 0.015
余额: 99.985 (初始100, 扣0.015)
节点: Online, 成功率: 100%, EMA延迟: 2788ms, EMA速度: 1.94 tok/s
```

## 待推进（不依赖 GPU）
1. **EvoAgent 接入 DecentralAI** — EvoAgent 作为节点客户端，测试自产自销闭环
2. **多节点 P2P** — Router 发现更多 Worker，形成真实网络
3. **合约完善** — DecentralAI.sol 的链上结算集成
4. **评估框架** — 通用 benchmark 框架，GPU 到手即跑

## 关键文件
- `src-rs/decentral-ai-core/src/router.rs` (17KB)
- `src-rs/decentral-ai-core/src/router_main.rs` (2.5KB)
- `src-rs/decentral-ai-core/src/lib.rs` (1.3KB)
- `src-rs/decentral-ai-core/target/release/router.exe` (5.4MB)
