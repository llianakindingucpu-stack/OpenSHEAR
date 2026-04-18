// DecentralAI - P2P Network Layer
// Uses raw TCP + length-prefixed JSON (no libp2p dependency)
// Port 9090: P2P gossip server
// Protocol: <4-byte-big-endian-length>\n<json>\n

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ===================== Constants =====================

pub const P2P_PORT: u16 = 9090;
pub const PROTOCOL_VERSION: &str = "0.1.0";

/// Wire protocol: length prefix before each JSON message
fn encode_msg<T: Serialize>(msg: &T) -> Vec<u8> {
    let json = serde_json::to_string(msg).unwrap();
    let mut buf = json.into_bytes();
    let len = (buf.len() as u32).to_be_bytes();
    let mut out = len.to_vec();
    out.append(&mut buf);
    out.push(b'\n');
    out
}

fn decode_msg<R: AsyncReadExt + Unpin>(r: &mut R) -> impl std::future::Future<Output = Result<String, std::io::Error>> + '_ {
    async move {
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf).await?;
        // read trailing newline
        r.read_exact(&mut [0u8]).await?;
        String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// ===================== Data Types =====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: String,
    pub address: String,   // IP or domain
    pub p2p_port: u16,    // P2P port (9090)
    pub http_port: u16,   // HTTP API port (8080)
    pub tier: String,     // L0-L4
    pub capabilities: Vec<String>,
    pub reputation: f64,
    pub version: String,
    pub last_seen: u64,   // Unix seconds
}

impl NetworkNode {
    pub fn new_local(id: String, tier: String, http_port: u16, p2p_port: u16) -> Self {
        Self {
            id,
            address: "127.0.0.1".to_string(),
            p2p_port,
            http_port,
            tier,
            capabilities: vec!["inference".to_string()],
            reputation: 50.0,
            version: PROTOCOL_VERSION.to_string(),
            last_seen: unix_ts(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WireMessage {
    // Discovery
    Ping { node: NetworkNode },
    Pong { node: NetworkNode },

    // Node registry
    Gossip { nodes: Vec<NetworkNode> },
    GetNodes,
    Nodes { nodes: Vec<NetworkNode> },

    // Inference delegation
    InferenceRequest {
        request_id: String,
        model: String,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
    },
    InferenceResponse {
        request_id: String,
        text: String,
        tokens: usize,
        from: String,
    },

    // Credits
    CreditNotice { from: String, to: String, amount: f64 },

    // Bootstrap
    Handshake { node: NetworkNode },
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// ===================== Shared State =====================

pub type SharedState = Arc<P2PState>;

pub struct P2PState {
    pub local_node: NetworkNode,
    peers: RwLock<HashMap<String, NetworkNode>>,
    #[allow(dead_code)]
    pending_requests: RwLock<HashMap<String, mpsc::Sender<WireMessage>>>,
    gossip_tx: broadcast::Sender<WireMessage>,
}

impl P2PState {
    pub fn new(local_node: NetworkNode) -> Self {
        let (gossip_tx, _) = broadcast::channel(1024);
        Self {
            local_node,
            peers: RwLock::new(HashMap::new()),
            pending_requests: RwLock::new(HashMap::new()),
            gossip_tx,
        }
    }

    pub async fn add_peer(&self, node: NetworkNode) {
        let mut peers = self.peers.write().await;
        peers.insert(node.id.clone(), node);
    }

    pub async fn get_peers(&self) -> Vec<NetworkNode> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    pub async fn remove_stale_peers(&self, max_age_secs: u64) {
        let now = unix_ts();
        let mut peers = self.peers.write().await;
        peers.retain(|_, n| now.saturating_sub(n.last_seen) < max_age_secs);
    }

    /// Connect to a peer via TCP
    pub async fn connect_peer(&self, addr: SocketAddr) -> Result<(), String> {
        let mut stream = TcpStream::connect(addr)
            .await
            .map_err(|e| format!("connect failed: {e}"))?;

        // Send handshake
        let msg = WireMessage::Handshake { node: self.local_node.clone() };
        stream
            .write_all(&encode_msg(&msg))
            .await
            .map_err(|e| format!("write failed: {e}"))?;

        // Read response
        let resp = decode_msg(&mut stream).await.map_err(|e| format!("read failed: {e}"))?;
        let reply: WireMessage = serde_json::from_str(&resp)
            .map_err(|e| format!("parse failed: {e}"))?;

        match reply {
            WireMessage::Handshake { node } => {
                self.add_peer(node).await;
                tracing::info!("[P2P] Connected to peer {}", addr);
            }
            _ => return Err("Expected handshake response".to_string()),
        }

        Ok(())
    }

    /// Broadcast a message to all connected peers
    pub async fn broadcast(&self, msg: WireMessage) {
        let peers = self.get_peers().await;
        let _node = self.local_node.clone();
        let gossip_tx = self.gossip_tx.clone();

        for peer in peers {
            let addr_str = format!("{}:{}", peer.address, peer.p2p_port);
            let addr: SocketAddr = match addr_str.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };

            let msg_clone = msg.clone();
            tokio::spawn(async move {
                if let Ok(mut stream) = TcpStream::connect(addr).await {
                    let _ = stream.write_all(&encode_msg(&msg_clone)).await;
                }
            });
        }

        // Also put on local gossip channel
        let _ = gossip_tx.send(msg);
    }
}

// ===================== P2P Server =====================

/// Start the P2P gossip server on a given port
pub async fn start_p2p_server(state: SharedState, port: u16) {
    let addr = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("[P2P] Failed to bind port {}: {e}", port);
            return;
        }
    };

    tracing::info!("[P2P] Server listening on {}", addr);

    loop {
        match listener.accept().await {
            Ok((stream, remote_addr)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_p2p_connection(stream, remote_addr, state).await {
                        tracing::debug!("[P2P] Connection error from {}: {e}", remote_addr);
                    }
                });
            }
            Err(e) => {
                tracing::error!("[P2P] Accept error: {e}");
            }
        }
    }
}

async fn handle_p2p_connection(
    mut stream: TcpStream,
    remote_addr: SocketAddr,
    state: SharedState,
) -> Result<(), String> {
    // Enable TCP keepalive
    stream
        .set_nodelay(true)
        .map_err(|e| format!("set_nodelay: {e}"))?;

    loop {
        // Read message with timeout
        let msg = match tokio::time::timeout(
            std::time::Duration::from_secs(60),
            decode_msg(&mut stream),
        )
        .await
        {
            Ok(Ok(m)) => m,
            Ok(Err(_)) => break,           // connection closed
            Err(_) => break,               // timeout
        };

        let wire: WireMessage = match serde_json::from_str(&msg) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("[P2P] Invalid message from {}: {e}", remote_addr);
                continue;
            }
        };

        // Process message
        let response = process_message(wire, &state).await;

        // Send response if any
        if let Some(resp) = response {
            stream
                .write_all(&encode_msg(&resp))
                .await
                .map_err(|e| format!("write error: {e}"))?;
        }
    }

    Ok(())
}

async fn process_message(msg: WireMessage, state: &SharedState) -> Option<WireMessage> {
    match msg {
        WireMessage::Ping { node } => {
            tracing::debug!("[P2P] Ping from {}", node.id);
            state.add_peer(node).await;
            Some(WireMessage::Pong { node: state.local_node.clone() })
        }

        WireMessage::Pong { node } => {
            state.add_peer(node).await;
            None
        }

        WireMessage::Handshake { node } => {
            tracing::info!("[P2P] Handshake with {}", node.id);
            state.add_peer(node).await;
            // Announce to existing peers
            state.broadcast(WireMessage::Gossip {
                nodes: vec![state.local_node.clone()],
            }).await;
            Some(WireMessage::Handshake { node: state.local_node.clone() })
        }

        WireMessage::Gossip { nodes } => {
            tracing::debug!("[P2P] Gossip: {} nodes", nodes.len());
            for node in nodes {
                state.add_peer(node).await;
            }
            None
        }

        WireMessage::GetNodes => {
            let nodes = state.get_peers().await;
            Some(WireMessage::Nodes { nodes })
        }

        WireMessage::Nodes { nodes } => {
            for node in nodes {
                state.add_peer(node).await;
            }
            None
        }

        WireMessage::InferenceRequest {
            request_id,
            model,
            prompt,
            max_tokens,
            temperature,
        } => {
            tracing::info!(
                "[P2P] Inference request {} -> model={} tokens={}",
                request_id, model, max_tokens
            );
            // Forward to local Python Worker
            let result = forward_to_worker(&model, &prompt, max_tokens, temperature).await;
            Some(WireMessage::InferenceResponse {
                request_id,
                text: result.clone(),
                tokens: result.split_whitespace().count(),
                from: state.local_node.id.clone(),
            })
        }

        WireMessage::InferenceResponse { .. } => {
            // TODO: resolve pending request
            None
        }

        WireMessage::CreditNotice { .. } => {
            // TODO: record to blockchain
            None
        }
    }
}

/// Forward inference request to local Python Worker (called by main.rs)
pub async fn forward_to_worker(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> String {
    let url = "http://127.0.0.1:8081/v1/chat/completions".to_string();

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    });

    // ureq is blocking, run in spawn_blocking to avoid blocking the async runtime
    let body_str = serde_json::to_string(&body).unwrap_or_default();
    let resp = match tokio::task::spawn_blocking(move || {
        ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body_str)
    }).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return format!("[worker error: {}]", e),
        Err(e) => return format!("[spawn blocked: {}]", e),
    };

    if !(200..300).contains(&resp.status()) {
        return format!("[worker HTTP {}]", resp.status());
    }

    match resp.into_string() {
        Ok(body) => {
            let json: serde_json::Value = match serde_json::from_str(&body) {
                Ok(j) => j,
                Err(_) => return "[parse error]".to_string(),
            };
            json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("[no content]")
                .to_string()
        }
        Err(_) => "[response read error]".to_string(),
    }
}

// ===================== Bootstrap =====================

/// Bootstrap: connect to known seed nodes and exchange peer lists
pub async fn bootstrap_peers(state: SharedState) {
    let seeds = vec![
        ("127.0.0.1", 9090u16), // local dev
    ];

    for (host, port) in seeds {
        let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();
        if addr == format!("0.0.0.0:{}", P2P_PORT).parse().unwrap() {
            continue; // skip self
        }
        if let Err(e) = state.connect_peer(addr).await {
            tracing::debug!("[P2P] Seed {} unavailable: {e}", addr);
        }
    }

    // Periodically send gossip
    let state2 = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let peers = state2.get_peers().await;
            if !peers.is_empty() {
                state2.broadcast(WireMessage::Gossip { nodes: peers }).await;
            }
        }
    });
}

// ===================== HTTP Bridge (P2P status via REST) =====================

use axum::{extract::State, Json, http::StatusCode};

/// P2P peers list endpoint (called from main.rs router)
pub async fn p2p_peers_handler(state: State<Arc<crate::AppState>>) -> Json<serde_json::Value> {
    let peers = state.p2p_state.get_peers().await;
    Json(serde_json::json!({
        "local_node": state.p2p_state.local_node,
        "peers": peers,
        "peer_count": peers.len(),
    }))
}

pub async fn p2p_connect_handler(
    state: State<Arc<crate::AppState>>,
    axum::extract::Path(addr): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use std::net::SocketAddr;
    let target: SocketAddr = addr.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    match state.p2p_state.connect_peer(target).await {
        Ok(_) => Ok(Json(serde_json::json!({ "status": "connected", "addr": addr }))),
        Err(e) => Ok(Json(serde_json::json!({ "status": "error", "message": e }))),
    }
}
