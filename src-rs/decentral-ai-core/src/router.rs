// DecentralAI Router — multi-node inference dispatcher
// Replaces reqwest with ureq (no rand dep) to avoid conflict with candle.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::{
    routing::{get, post, delete},
    extract::{Path, Query, State},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use ureq::Agent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus { Online, Offline, Unhealthy }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub url: String,
    pub p2p_addr: String,
    pub level: u8,
    pub model: String,
    pub status: NodeStatus,
    pub avg_latency_ms: f64,
    pub avg_tokens_per_sec: f64,
    pub success_rate: f64,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub last_heartbeat: f64,
    pub registered_at: f64,
}

impl NodeInfo {
    pub fn new(node_id: String, url: String, p2p_addr: String, level: u8, model: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        Self { node_id, url, p2p_addr, level, model,
            status: NodeStatus::Online,
            avg_latency_ms: 2000.0, avg_tokens_per_sec: 0.0, success_rate: 0.0,
            total_requests: 0, failed_requests: 0,
            last_heartbeat: now, registered_at: now }
    }

    pub fn score(&self) -> f64 {
        if self.status != NodeStatus::Online { return 0.0; }
        let ss = self.success_rate;
        let sp = (self.avg_tokens_per_sec / 20.0).min(1.0);
        let ls = 1.0 - (self.avg_latency_ms / 5000.0).min(1.0);
        let gs = (self.level as f64) / 4.0;
        ss * 0.4 + sp * 0.3 + ls * 0.2 + gs * 0.1
    }

    pub fn update_metrics(&mut self, latency_ms: f64, tokens: u32, success: bool) {
        let a = 0.3;
        self.total_requests += 1;
        if !success { self.failed_requests += 1; }
        self.avg_latency_ms = a * latency_ms + (1.0 - a) * self.avg_latency_ms;
        if latency_ms > 0.0 && tokens > 0 {
            let tps = tokens as f64 / (latency_ms / 1000.0);
            self.avg_tokens_per_sec = a * tps + (1.0 - a) * self.avg_tokens_per_sec;
        }
        self.success_rate = 1.0 - (self.failed_requests as f64) / (self.total_requests as f64).max(1.0);
    }
}

fn credits_rate(model: &str) -> f64 {
    match model {
        "rwkv-4-169m" => 0.5, "rwkv-4-430m" => 0.8,
        "qwen-0.5b" => 1.0, "qwen-1.5b" => 1.5, "qwen-7b" => 2.5,
        _ => 1.0,
    }
}

// ── Router State ───────────────────────────────────────────

#[derive(Clone)]
pub struct RouterState {
    pub nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    pub http_agent: Agent,
    pub db_path: String,
}

impl RouterState {
    pub fn new(db_path: String) -> Self {
        let agent = Agent::new();
        Self { nodes: Arc::new(RwLock::new(HashMap::new())), http_agent: agent, db_path }
    }

    pub fn register(&self, node: NodeInfo) {
        self.nodes.write().unwrap().insert(node.node_id.clone(), node);
    }
    pub fn unregister(&self, node_id: &str) {
        self.nodes.write().unwrap().remove(node_id);
    }

    pub fn get_best_nodes(&self, model: &str, count: usize) -> Vec<(String, f64)> {
        let nodes = self.nodes.read().unwrap();
        let mut cand: Vec<_> = nodes.values()
            .filter(|n| n.status == NodeStatus::Online && (n.model == model || model.is_empty()))
            .map(|n| (n.node_id.clone(), n.score()))
            .collect();
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        cand.truncate(count);
        cand
    }

    pub fn get_best_node(&self, model: &str) -> Option<(String, NodeInfo)> {
        self.get_best_nodes(model, 1)
            .first()
            .and_then(|(id, _)| self.nodes.read().unwrap().get(id).cloned().map(|n| (id.clone(), n)))
    }

    pub fn heartbeat(&self, node_id: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        let mut ns = self.nodes.write().unwrap();
        if let Some(n) = ns.get_mut(node_id) {
            n.last_heartbeat = now;
            if !matches!(n.status, NodeStatus::Online) { n.status = NodeStatus::Online; }
        }
    }

    pub fn get_all_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.read().unwrap().values().cloned().collect()
    }

    pub fn stats(&self) -> RouterStats {
        let ns = self.nodes.read().unwrap();
        let total = ns.len();
        let online = ns.values().filter(|n| matches!(n.status, NodeStatus::Online)).count();
        RouterStats { total_nodes: total, online_nodes: online }
    }

    pub fn health_check_node(&self, node_id: &str) -> bool {
        let url = self.nodes.read().unwrap().get(node_id).map(|n| n.url.clone());
        match url {
            Some(u) => self.http_agent.get(&format!("{}/health", u)).call().is_ok(),
            None => false,
        }
    }

    pub fn execute_on_node(
        &self, node_id: &str, prompt: &str, model: &str, max_tokens: usize, temperature: f32,
    ) -> Result<ExecuteResult, String> {
        let url = {
            let ns = self.nodes.read().unwrap();
            ns.get(node_id).map(|n| n.url.clone())
        }.ok_or("Node not found")?;

        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        });
        let body_str = serde_json::to_string(&body).unwrap_or_default();

        let start = Instant::now();
        let resp = self.http_agent
            .post(&format!("{}/v1/chat/completions", url))
            .set("Content-Type", "application/json")
            .send_string(&body_str)
            .map_err(|e| format!("ureq send: {}", e))?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !(200..300).contains(&resp.status()) {
            let body_err = resp.status_text().to_string();
            return Err(format!("Worker {} HTTP {}: {}", node_id, resp.status(), body_err));
        }

        let body_str = resp.into_string().map_err(|e| format!("read body: {}", e))?;
        let data: serde_json::Value = serde_json::from_str(&body_str)
            .map_err(|e| format!("JSON parse: {} | body: {}", e, &body_str[..body_str.len().min(200)]))?;

        let content = data["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
        let tokens = data["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;

        { // Update metrics
            let mut ns = self.nodes.write().unwrap();
            if let Some(n) = ns.get_mut(node_id) { n.update_metrics(latency_ms, tokens, true); }
        }

        Ok(ExecuteResult { node_id: node_id.to_string(), text: content, tokens, latency_ms })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteResult { pub node_id: String, pub text: String, pub tokens: u32, pub latency_ms: f64 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats { pub total_nodes: usize, pub online_nodes: usize }

// ── SQLite Credits ──────────────────────────────────────────

use rusqlite::{Connection, OpenFlags, params};

impl RouterState {
    fn get_conn(&self) -> Connection {
        if self.db_path == ":memory:" {
            Connection::open_in_memory().expect("in-memory db")
        } else {
            Connection::open_with_flags(&self.db_path, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE)
                .expect("open router.db")
        }
    }

    pub fn init_db(&self) {
        let conn = self.get_conn();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS credits (user_id TEXT PRIMARY KEY, balance REAL DEFAULT 100.0);
             CREATE TABLE IF NOT EXISTS credits_log (id INTEGER PRIMARY KEY, user_id TEXT, amount REAL, reason TEXT, task_id TEXT, timestamp INTEGER);
             CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, user_id TEXT, model TEXT, prompt TEXT, max_tokens INTEGER, status TEXT, result TEXT, latency_ms REAL, credits_cost REAL, nodes_used INTEGER, created_at INTEGER);"
        ).ok();
    }

    pub fn deduct_credits(&self, user_id: &str, amount: f64, reason: &str, task_id: &str) {
        let conn = self.get_conn();
        conn.execute("INSERT OR IGNORE INTO credits (user_id) VALUES (?)", params![user_id]).ok();
        conn.execute("UPDATE credits SET balance = balance - ? WHERE user_id = ?", params![amount, user_id]).ok();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        conn.execute("INSERT INTO credits_log (user_id, amount, reason, task_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            params![user_id, -amount, reason, task_id, now]).ok();
    }

    pub fn refund_credits(&self, user_id: &str, amount: f64, reason: &str, task_id: &str) {
        let conn = self.get_conn();
        conn.execute("UPDATE credits SET balance = balance + ? WHERE user_id = ?", params![amount, user_id]).ok();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        conn.execute("INSERT INTO credits_log (user_id, amount, reason, task_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            params![user_id, amount, reason, task_id, now]).ok();
    }

    pub fn get_balance(&self, user_id: &str) -> f64 {
        self.get_conn().query_row("SELECT balance FROM credits WHERE user_id = ?", params![user_id], |r| r.get::<_, f64>(0))
            .unwrap_or(100.0)
    }

    pub fn record_task(&self, task_id: &str, user_id: &str, model: &str, prompt: &str,
                       max_tokens: usize, status: &str, result: &str, latency_ms: f64, cost: f64, nodes_used: usize) {
        let conn = self.get_conn();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, user_id, model, prompt, max_tokens, status, result, latency_ms, credits_cost, nodes_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![task_id, user_id, model, prompt, max_tokens as i64, status, result, latency_ms, cost, nodes_used as i64, now]
        ).ok();
    }
}

// ── HTTP Handlers ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RegisterReq { pub node_id: String, pub url: String, pub p2p_addr: Option<String>, pub level: Option<u8>, pub model: Option<String> }

#[derive(Debug, Deserialize)]
pub struct RouteReq {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)] pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    #[serde(default = "anon")] pub user_id: String,
    #[serde(default = "one")] pub redundancy: usize,
}
fn anon() -> String { "anonymous".to_string() }
fn one() -> usize { 1 }

async fn list_nodes(State(s): State<RouterState>) -> Json<NodeListResp> {
    Json(NodeListResp { count: s.get_all_nodes().len(), nodes: s.get_all_nodes() })
}

async fn register_node(State(s): State<RouterState>, Json(req): Json<RegisterReq>) -> Json<serde_json::Value> {
    let node = NodeInfo::new(req.node_id, req.url, req.p2p_addr.unwrap_or_default(),
        req.level.unwrap_or(1), req.model.unwrap_or_else(|| "unknown".to_string()));
    s.register(node);
    Json(serde_json::json!({"status": "registered"}))
}

async fn unregister_node(State(s): State<RouterState>, Path(node_id): Path<String>) -> Json<serde_json::Value> {
    s.unregister(&node_id);
    Json(serde_json::json!({"status": "unregistered"}))
}

async fn heartbeat(State(s): State<RouterState>, Json(payload): Json<serde_json::Value>) -> Json<serde_json::Value> {
    if let Some(id) = payload["node_id"].as_str() { s.heartbeat(id); Json(serde_json::json!({"status": "ok"})) }
    else { Json(serde_json::json!({"status": "error"})) }
}

async fn stats(State(s): State<RouterState>) -> Json<RouterStats> { Json(s.stats()) }

async fn balance(State(s): State<RouterState>, Query(q): Query<HashMap<String, String>>) -> Json<serde_json::Value> {
    let uid = q.get("user_id").cloned().unwrap_or_else(|| "anonymous".to_string());
    Json(serde_json::json!({"user_id": uid, "balance": s.get_balance(&uid)}))
}

async fn forward(State(s): State<RouterState>, Json(req): Json<RouteReq>) -> Json<serde_json::Value> {
    let task_id = format!("task-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
    let model = if req.model.is_empty() { "rwkv-4-169m" } else { &req.model };
    let max_tokens = req.max_tokens.unwrap_or(128).min(512);
    let temperature = req.temperature.unwrap_or(0.7);
    let prompt = if !req.messages.is_empty() {
        req.messages.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n")
    } else { req.prompt.clone() };

    let nodes = s.get_best_nodes(model, req.redundancy.max(1));
    if nodes.is_empty() {
        return Json(serde_json::json!({"error": {"message": "No available nodes", "type": "router_error"}}));
    }

    let node_id = &nodes[0].0;
    let estimated = credits_rate(model) * (max_tokens as f64) / 1000.0;
    s.deduct_credits(&req.user_id, estimated, "pre-auth", &task_id);

    let start = Instant::now();
    match s.execute_on_node(node_id, &prompt, model, max_tokens, temperature) {
        Ok(result) => {
            let actual = credits_rate(model) * (result.tokens as f64) / 1000.0;
            if estimated > actual { s.refund_credits(&req.user_id, estimated - actual, "refund", &task_id); }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            s.record_task(&task_id, &req.user_id, model, &prompt, max_tokens, "completed", &result.text, ms, actual, 1);
            Json(serde_json::json!({"task_id": task_id, "text": result.text, "tokens": result.tokens,
                "latency_ms": ms, "credits_cost": actual, "nodes_used": 1, "node_id": node_id}))
        }
        Err(e) => {
            s.refund_credits(&req.user_id, estimated, &format!("failed: {}", e), &task_id);
            s.record_task(&task_id, &req.user_id, model, &prompt, max_tokens, "failed", "", 0.0, 0.0, 0);
            Json(serde_json::json!({"error": {"message": e, "type": "execution_error"}}))
        }
    }
}

async fn router_health(State(s): State<RouterState>) -> Json<serde_json::Value> {
    let st = s.stats();
    Json(serde_json::json!({"status": "ok", "role": "router",
        "total_nodes": st.total_nodes, "online_nodes": st.online_nodes}))
}

#[derive(Debug, Serialize)]
pub struct NodeListResp { pub nodes: Vec<NodeInfo>, pub count: usize }

// ── Health Checker ──────────────────────────────────────────

pub async fn start_health_checker(state: RouterState) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            let ids: Vec<String> = { state.nodes.read().unwrap().keys().cloned().collect() };
            for id in ids {
                let healthy = state.health_check_node(&id);
                let mut ns = state.nodes.write().unwrap();
                if let Some(n) = ns.get_mut(&id) {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
                    if now - n.last_heartbeat > 30.0 {
                        n.status = if healthy { NodeStatus::Online } else { NodeStatus::Unhealthy };
                    }
                }
            }
        }
    });
}

// ── Build App ──────────────────────────────────────────────

pub fn build_router_app(state: RouterState) -> Router {
    Router::new()
        .route("/health", get(router_health))
        .route("/nodes", get(list_nodes))
        .route("/register", post(register_node))
        .route("/unregister/:node_id", delete(unregister_node))
        .route("/heartbeat", post(heartbeat))
        .route("/stats", get(stats))
        .route("/balance", get(balance))
        .route("/forward", post(forward))
        .with_state(state)
}
