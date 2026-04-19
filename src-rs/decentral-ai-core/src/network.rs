//! SHEAR P2P Network Layer
//!
//! ## Architecture
//!
//! - **Kademlia DHT**: peer discovery + node record storage (key = PeerId bytes)
//! - **mDNS**: zero-config LAN peer discovery
//! - **Gossipsub**: heartbeat broadcast, consensus requests, verified results
//!
//! ## Topics
//!
//! - `shear/heartbeat/1` — Node heartbeat (tier, availability, reputation)
//! - `shear/consensus/1` — Inference requests needing multi-node verification
//! - `shear/verified/1` — Verified inference results
//!
//! ## Planned Implementation
//!
//! ```ignore
//! use decentral_ai_core::network::{P2PNode, NodeTier, P2PEvent};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let (mut node, mut events) = P2PNode::new(
//!         NodeTier::L2,
//!         "/ip4/0.0.0.0/tcp/9090",
//!     ).await?;
//!     node.dial("12D3KooWExampleBoostrapPeerId".parse()?).await?;
//!     tokio::spawn(async move {
//!         while let Some(event) = events.recv().await {
//!             match event {
//!                 P2PEvent::PeerDiscovered { peer_id, tier } => {
//!                     tracing::info!("Discovered peer {peer_id:?} tier {tier:?}");
//!                 }
//!                 P2PEvent::ConsensusRequestReceived(req) => {
//!                     // Process request, broadcast result
//!                 }
//!                 P2PEvent::HeartbeatReceived(hb) => {
//!                     // Update peer health
//!                 }
//!                 _ => {}
//!             }
//!         }
//!     });
//!     node.run().await;
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::extract::State;
use axum::response::Json;
use tokio::sync::{broadcast, RwLock};

/// ---------------------------------------------------------------------------
/// Types
/// ---------------------------------------------------------------------------

/// SHEAR node tier. Matches the SHEAR tier system (L0-L4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeTier {
    /// Collector — CPU only, no model inference
    L0,
    /// Light inference — 0.5B-1.5B model
    L1,
    /// Standard — 7B model, dedicated GPU (RTX 3060+)
    L2,
    /// Heavy — 14B+ model, high-end GPU (RTX 4090)
    L3,
    /// Datacenter — 70B+ model, A100/H100 cluster
    L4,
}

impl std::fmt::Display for NodeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeTier::L0 => write!(f, "l0"),
            NodeTier::L1 => write!(f, "l1"),
            NodeTier::L2 => write!(f, "l2"),
            NodeTier::L3 => write!(f, "l3"),
            NodeTier::L4 => write!(f, "l4"),
        }
    }
}

/// Node record stored in Kademlia DHT. Peers look this up by PeerId.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRecord {
    /// Base58-encoded PeerId
    pub peer_id: String,
    /// Node tier
    pub tier: NodeTier,
    /// Price per 1K tokens in Credits
    pub credits_per_1k: u64,
    /// Whether the node is currently accepting requests
    pub available: bool,
    /// Reputation score 0.0-1.0
    pub reputation: f32,
    /// Last heartbeat unix timestamp
    pub last_seen: u64,
}

impl NodeRecord {
    pub fn new(peer_id: &str, tier: NodeTier) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            tier,
            credits_per_1k: 1000,
            available: true,
            reputation: 1.0,
            last_seen: unix_ts(),
        }
    }
}

/// Heartbeat broadcast payload (gossipsub shear/heartbeat/1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub peer_id: String,
    pub tier: NodeTier,
    pub available: bool,
    pub reputation: f32,
    pub credits_per_1k: u64,
    pub timestamp_secs: u64,
}

impl Heartbeat {
    pub fn now(peer_id: &str, tier: NodeTier) -> Self {
        Self {
            peer_id: peer_id.to_string(),
            tier,
            available: true,
            reputation: 1.0,
            credits_per_1k: 1000,
            timestamp_secs: unix_ts(),
        }
    }
}

/// Inference request needing multi-node consensus (gossipsub shear/consensus/1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request ID for deduplication
    pub request_id: String,
    /// Tokenized prompt
    pub prompt: String,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0 = greedy)
    pub temperature: f32,
    /// Target tier for this request (None = any tier)
    pub target_tier: Option<NodeTier>,
    /// PeerId of the original requestor
    pub source_peer: String,
    /// Hop counter (max 8 hops to prevent infinite routing)
    pub hop: u8,
}

impl InferenceRequest {
    /// Increment hop counter. Returns false if max hops exceeded.
    pub fn hop_forward(&mut self) -> bool {
        self.hop += 1;
        self.hop < 8
    }
}

/// Verified inference result (gossipsub shear/verified/1).
/// Broadcast after result has passed consensus verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub request_id: String,
    pub peer_id: String,
    pub tokens: Vec<u32>,
    /// Wall-clock time in ms
    pub latency_ms: u64,
    /// Credits charged
    pub credits_charged: u64,
}

/// Events emitted from P2P node to the application layer.
#[derive(Debug, Clone)]
pub enum P2PEvent {
    /// New peer discovered via DHT or mDNS.
    PeerDiscovered { peer_id: String, tier: NodeTier },
    /// Peer disconnected or timed out.
    PeerLost { peer_id: String },
    /// Heartbeat received from a peer.
    HeartbeatReceived(Heartbeat),
    /// Consensus inference request received.
    ConsensusRequestReceived(InferenceRequest),
    /// Verified inference result received.
    VerifiedResultReceived(InferenceResult),
}

/// ---------------------------------------------------------------------------
/// Shared State
/// ---------------------------------------------------------------------------

/// Shared peer table. Thread-safe, shared between P2P layer and HTTP router.
pub type SharedState = Arc<RwLock<HashMap<String, NodeRecord>>>;

/// Create a new empty shared peer state map.
pub fn new_shared_state() -> SharedState {
    Arc::new(RwLock::new(HashMap::new()))
}

/// ---------------------------------------------------------------------------
/// P2PNode
/// ---------------------------------------------------------------------------

/// SHEAR P2P Node.
///
/// Full implementation wraps a libp2p v0.56 Swarm with:
/// - Kademlia DHT: /shear/kad/1 protocol, peer discovery, record storage
/// - mDNS: zero-config LAN discovery
/// - Gossipsub: heartbeat + consensus + verified topics
///
/// # Status
///
/// **Stub** — type fully defined with all methods, comments, and planned API.
/// Full libp2p integration deferred until the P2P stack can be tested in
/// a real multi-node network environment.
///
///
/// ## Full Implementation Notes
///
/// ```
/// // Transport: TCP + noise (encryption) + yamux (stream mux)
/// let tcp = tcp::tokio::Transport::default();
/// let noise = noise::Config::new(keypair)?;
/// let transport = tcp.upgrade(V1).authenticate(noise).multiplex(yamux).boxed();
///
/// // Swarm with NetworkBehaviour (derived or manual)
/// let behaviour = SHEARBehaviour::new(local_peer_id);
/// behaviour.subscribe(); // gossip topics
/// let swarm = Swarm::new(transport, behaviour, local_peer_id, Config::with_tokio_executor());
/// swarm.listen_on(addr.parse()?)?;
///
/// // Main loop
/// loop {
///     tokio::select! {
///         event = swarm.next() => match event {
///             SwarmEvent::Behaviour(SHEARBehaviourEvent::Kad(e)) => handle_kad(e),
///             SwarmEvent::Behaviour(SHEARBehaviourEvent::Mdns(e)) => handle_mdns(e),
///             SwarmEvent::Behaviour(SHEARBehaviourEvent::Gossipsub(e)) => handle_gossip(e),
///         },
///         _ = heartbeat_interval.tick() => { broadcast_heartbeat().await; }
///     }
/// }
/// ```
pub struct P2PNode {
    /// Local PeerId (full impl: from libp2p identity::Keypair)
    pub peer_id: String,
    /// This node's tier
    pub tier: NodeTier,
    /// Shared peer table
    pub peers: SharedState,
    /// Event broadcast channel
    tx: broadcast::Sender<P2PEvent>,
}

impl P2PNode {
    /// Create a new P2P node listening on `listen_addr`.
    ///
    /// Full impl:
    /// 1. Generate libp2p::identity::Keypair → PeerId
    /// 2. Build TCP + noise + yamux transport
    /// 3. Create SHEARBehaviour (Kademlia + mDNS + Gossipsub)
    /// 4. Subscribe to gossip topics
    /// 5. Bind to listen_addr
    pub async fn new(tier: NodeTier, listen_addr: &str) -> anyhow::Result<(Self, broadcast::Receiver<P2PEvent>)> {
        let (tx, rx) = broadcast::channel(256);
        // Full impl: PeerId = from keypair.public()
        let peer_id = generate_peer_id();
        tracing::info!(
            "[P2P] Node {} starting as tier {} on {} [STUB - full impl uses libp2p]",
            peer_id, tier, listen_addr
        );
        let peers = new_shared_state();
        Ok((Self { peer_id: peer_id.clone(), tier, peers, tx }, rx))
    }

    /// Dial a peer at the given address.
    ///
    /// Full impl: Swarm::dial(multiaddr.parse()?)
    pub async fn dial(&mut self, addr: &str) -> anyhow::Result<()> {
        tracing::debug!("[P2P] dial {} [STUB]", addr);
        Ok(())
    }

    /// Bootstrap to a known seed node (Kademlia).
    pub async fn bootstrap(&mut self, peer_id: &str, addr: &str) -> anyhow::Result<()> {
        tracing::debug!("[P2P] bootstrap to {} at {} [STUB]", peer_id, addr);
        Ok(())
    }

    /// Publish our node record to the Kademlia DHT.
    ///
    /// Full impl: Kademlia::put_record(Record { key: PeerId bytes, value: NodeRecord JSON })
    pub async fn publish_record(&mut self) -> anyhow::Result<()> {
        let record = NodeRecord::new(&self.peer_id, self.tier);
        self.peers.write().await.insert(self.peer_id.clone(), record);
        tracing::debug!("[P2P] published node record [STUB]");
        Ok(())
    }

    /// Broadcast heartbeat via Gossipsub shear/heartbeat/1.
    ///
    /// Full impl: Gossipsub::publish(topic, serde_json::to_vec(Heartbeat))
    pub async fn send_heartbeat(&mut self) -> anyhow::Result<()> {
        let _hb = Heartbeat::now(&self.peer_id, self.tier);
        tracing::trace!("[P2P] heartbeat broadcast [STUB]");
        Ok(())
    }

    /// Broadcast an inference request needing multi-node consensus.
    ///
    /// Receiving peers process the request and broadcast the verified result.
    pub async fn broadcast_request(&mut self, req: &InferenceRequest) -> anyhow::Result<()> {
        tracing::debug!(
            "[P2P] broadcasting consensus request {} hop={} [STUB]",
            req.request_id, req.hop
        );
        Ok(())
    }

    /// Broadcast a verified inference result on shear/verified/1.
    pub async fn broadcast_result(&mut self, result: &InferenceResult) -> anyhow::Result<()> {
        tracing::debug!("[P2P] broadcasting verified result {} [STUB]", result.request_id);
        Ok(())
    }

    /// Get all known peer IDs of a specific tier.
    pub async fn peers_of_tier(&self, tier: NodeTier) -> Vec<String> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|r| r.tier == tier && r.available)
            .map(|r| r.peer_id.clone())
            .collect()
    }

    /// Get all known peer IDs.
    pub async fn all_peers(&self) -> Vec<String> {
        let peers = self.peers.read().await;
        peers.keys().cloned().collect()
    }

    /// Main P2P event loop.
    ///
    /// Full implementation:
    /// - Poll libp2p::Swarm for behaviour events
    /// - Every 30s: broadcast heartbeat (Gossipsub)
    /// - Every 5min: re-publish DHT record
    /// - Emit P2PEvent to application channel
    pub async fn run(&mut self) {
        tracing::info!("[P2P] Node {} entering event loop [STUB - full impl uses libp2p]", self.peer_id);
        let _ = self.publish_record().await;
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let _ = self.send_heartbeat().await;
                    // Full impl: poll Swarm here for real P2P behaviour events
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Current Unix timestamp in seconds.
fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Generate a plausible-looking PeerId string.
/// Full impl: PeerId = libp2p::identity::Keypair::generate_ed25519().public().to_bytes()
fn generate_peer_id() -> String {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(42_u128);
    format!(
        "12D3KooW{:016x}{:016x}",
        (seed >> 64) as u64,
        seed as u64
    )
}

// ============================================================================
// Backward-compatibility shim (for main.rs / existing binary targets)
// These symbols existed in the previous network.rs implementation.
// They are thin stubs that keep main.rs compiling while the real P2P stack
// is reimplemented with libp2p v0.56.
// ============================================================================

/// Default P2P listen port (used by main.rs)
pub const P2P_PORT: u16 = 9090;

/// Node info used by main.rs (simpler than NodeRecord).
#[derive(Debug, Clone)]
pub struct NetworkNode {
    pub node_id: String,
    pub role: String,
    pub http_port: u16,
    pub p2p_port: u16,
    pub available: bool,
    pub reputation: f32,
}

impl NetworkNode {
    /// Create a local node entry.
    pub fn new_local(node_id: String, role: String, http_port: u16, p2p_port: u16) -> Self {
        Self { node_id, role, http_port, p2p_port, available: true, reputation: 1.0 }
    }
}

/// P2P state container used by main.rs.
/// In the full implementation, this holds the libp2p Swarm.
#[derive(Clone)]
pub struct P2PState {
    pub local_node: Arc<NetworkNode>,
}

impl P2PState {
    pub fn new(local_node: NetworkNode) -> Self {
        Self { local_node: Arc::new(local_node) }
    }
}

/// Forward an inference request to the Python worker (stub).
/// Full impl: HTTP POST to Python inference_worker.py :8081.
pub async fn forward_to_worker(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> String {
    tracing::debug!(
        "[P2P] forward_to_worker model={} prompt_len={} max_tokens={} temp={} [STUB]",
        model,
        prompt.len(),
        max_tokens,
        temperature
    );
    // Full impl: POST to http://127.0.0.1:8081/inference with JSON body
    // and return the "text" field from the response.
    // For now, return a stub response.
    format!(
        "[stub inference for model '{}', {} chars input, {} tokens, temp {}]",
        model,
        prompt.len(),
        max_tokens,
        temperature
    )
}

/// Start the P2P server (stub).
/// Full impl: bind TCP :p2p_port, run libp2p Swarm event loop.
pub async fn start_p2p_server(_state: Arc<P2PState>, _port: u16) {
    tracing::info!("[P2P] start_p2p_server on port {} [STUB - full impl uses libp2p]", _port);
    // Full impl: tokio::spawn(P2PNode::new(...).run().await)
    // Keep-alive: loop { tokio::time::sleep(Duration::MAX).await; }
    std::future::pending::<()>().await;
}

/// Bootstrap to hardcoded seed peers (stub).
pub async fn bootstrap_peers(_state: Arc<P2PState>) {
    tracing::info!("[P2P] bootstrap_peers [STUB - no seed peers configured]");
    // Full impl: for each seed addr in config, state.p2p_node.dial(addr).await
}

/// HTTP handler: GET /p2p/peers
/// Compatible with AppState (used by main.rs via with_state).
pub async fn p2p_peers_handler(
    State(state): State<Arc<crate::AppState>>,
) -> Json<serde_json::Value> {
    let local = &state.p2p_state.local_node;
    Json(serde_json::json!({
        "node_id": local.node_id,
        "role": local.role,
        "p2p_port": local.p2p_port,
        "available": local.available,
        "reputation": local.reputation,
        "credits": state.credits,
        "reputation_app": state.reputation,
        "peers": [],  // full impl: state.p2p_node.all_peers().await
    }))
}

/// HTTP handler: POST /p2p/connect/:addr
/// Compatible with AppState.
pub async fn p2p_connect_handler(
    State(_state): State<Arc<crate::AppState>>,
    axum::extract::Path(addr): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    tracing::info!("[P2P] connect request to {} [STUB]", addr);
    Json(serde_json::json!({ "status": "ok", "message": "STUB - P2P connection requested", "addr": addr }))
}
