// DecentralAI Core - Main entry point
// Modules are declared in lib.rs, imported from the crate

use std::net::SocketAddr;
use std::sync::Arc;
use axum::{routing::{get, post}, Router, Json, extract::State};
use serde::{Deserialize, Serialize};
use tracing_subscriber;

use decentral_ai_core::{
    AppState, NodeRole, ChatMessage,
    network,
};

// ==================== API Data Structures ====================

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens: usize,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub role: NodeRole,
    pub capabilities: Vec<String>,
    pub reputation: f64,
    pub url: String,
    pub p2p_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditBalance {
    pub node_id: String,
    pub balance: f64,
    pub reputation: f64,
}

// ==================== HTTP Handlers ====================

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "decentral-ai-core",
        "version": "0.1.0",
        "p2p_port": network::P2P_PORT,
    }))
}

async fn chat_completions(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    let max_tokens = req.max_tokens.unwrap_or(128);
    let temperature = req.temperature.unwrap_or(0.7);

    let text_input = if !req.messages.is_empty() {
        req.messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        req.prompt.clone()
    };

    let text = network::forward_to_worker(&req.model, &text_input, max_tokens, temperature).await;
    let tokens = text.split_whitespace().count();

    Json(InferenceResponse { text, tokens, model: req.model })
}

async fn node_info(State(state): State<Arc<AppState>>) -> Json<NodeInfo> {
    Json(NodeInfo {
        id: state.node_id.clone(),
        role: state.role.clone(),
        capabilities: vec!["inference".to_string()],
        reputation: state.reputation,
        url: "http://127.0.0.1:8080".to_string(),
        p2p_port: state.p2p_state.local_node.p2p_port,
    })
}

async fn credits(State(state): State<Arc<AppState>>) -> Json<CreditBalance> {
    Json(CreditBalance {
        node_id: state.node_id.clone(),
        balance: state.credits,
        reputation: state.reputation,
    })
}

// ==================== Main ====================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    tracing::info!("========================================");
    tracing::info!("  DecentralAI Core v0.1.0");
    tracing::info!("========================================");

    let args: Vec<String> = std::env::args().collect();
    let node_id = args.get(1).cloned().unwrap_or_else(|| "node-001".to_string());
    let role_str = args.get(2).cloned().unwrap_or_else(|| "L1".to_string());
    let http_port: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8080);
    let p2p_port: u16 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(network::P2P_PORT);

    let role = match role_str.as_str() {
        "L0" => NodeRole::L0Collector,
        "L1" => NodeRole::L1Lightweight,
        "L2" => NodeRole::L2Standard,
        "L3" => NodeRole::L3Heavy,
        "L4" => NodeRole::L4Datacenter,
        _ => NodeRole::L1Lightweight,
    };

    tracing::info!("Node ID:  {}", node_id);
    tracing::info!("Role:     {:?}", role);
    tracing::info!("HTTP:     127.0.0.1:{}", http_port);
    tracing::info!("P2P:      127.0.0.1:{}", p2p_port);

    let local_node = network::NetworkNode::new_local(
        node_id.clone(), role.to_string(), http_port, p2p_port,
    );
    let p2p_state = Arc::new(network::P2PState::new(local_node));
    let app_state = Arc::new(AppState::new(node_id.clone(), role, p2p_state.clone()));

    let p2p_state_spawn = p2p_state.clone();
    tokio::spawn(async move {
        network::start_p2p_server(p2p_state_spawn, p2p_port).await;
    });

    let p2p_state_bootstrap = p2p_state.clone();
    tokio::spawn(async move {
        network::bootstrap_peers(p2p_state_bootstrap).await;
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(node_info))
        .route("/credits", get(credits))
        .route("/p2p/peers", get(network::p2p_peers_handler))
        .route("/p2p/connect/:addr", post(network::p2p_connect_handler))
        .with_state(app_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], http_port));
    tracing::info!("========================================");
    tracing::info!("HTTP API:  http://{}", addr);
    tracing::info!("P2P:       127.0.0.1:{}", p2p_port);
    tracing::info!("========================================");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
