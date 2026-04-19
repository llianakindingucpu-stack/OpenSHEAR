//! SHEAR P2P Network Layer — libp2p v0.56
//!
//! ## Architecture
//!
//! - **Kademlia DHT**: peer discovery + record storage (primary stream)
//! - **mDNS**: zero-config LAN discovery (spawned polling task)
//! - **Gossipsub**: heartbeat, consensus, verified topics (main loop)
//!
//! ## Topics
//!
//! - `shear/heartbeat/1` — Node heartbeat
//! - `shear/consensus/1` — Inference consensus requests
//! - `shear/verified/1` — Verified inference results

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::Json;
use libp2p::{
    gossipsub, kad, mdns,
    identity::Keypair,
    PeerId, Swarm, Transport,
    swarm::NetworkBehaviour,
};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeTier { L0, L1, L2, L3, L4 }

impl NodeTier {
    pub fn from_role(role: &str) -> Self {
        match role {
            "L0" | "l0" => NodeTier::L0,
            "L1" | "l1" => NodeTier::L1,
            "L2" | "l2" => NodeTier::L2,
            "L3" | "l3" => NodeTier::L3,
            "L4" | "l4" => NodeTier::L4,
            _ => NodeTier::L1,
        }
    }
}

impl std::fmt::Display for NodeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self { NodeTier::L0 => "l0", NodeTier::L1 => "l1",
            NodeTier::L2 => "l2", NodeTier::L3 => "l3", NodeTier::L4 => "l4" };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRecord {
    pub peer_id: String,
    pub tier: NodeTier,
    pub credits_per_1k: u64,
    pub available: bool,
    pub reputation: f32,
    pub last_seen: u64,
}

impl NodeRecord {
    pub fn new(peer_id: &str, tier: NodeTier) -> Self {
        Self { peer_id: peer_id.to_string(), tier, credits_per_1k: 1000,
            available: true, reputation: 1.0, last_seen: unix_ts() }
    }
}

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
        Self { peer_id: peer_id.to_string(), tier, available: true,
            reputation: 1.0, credits_per_1k: 1000, timestamp_secs: unix_ts() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub target_tier: Option<NodeTier>,
    pub source_peer: String,
    pub hop: u8,
}

impl InferenceRequest {
    pub fn hop_forward(&mut self) -> bool { self.hop += 1; self.hop < 8 }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub request_id: String,
    pub peer_id: String,
    pub tokens: Vec<u32>,
    pub latency_ms: u64,
    pub credits_charged: u64,
}

#[derive(Debug, Clone)]
pub enum P2PEvent {
    PeerDiscovered { peer_id: String, tier: NodeTier },
    PeerLost { peer_id: String },
    HeartbeatReceived(Heartbeat),
    ConsensusRequestReceived(InferenceRequest),
    VerifiedResultReceived(InferenceResult),
}

// ---------------------------------------------------------------------------
// Shared State
// ---------------------------------------------------------------------------

pub type SharedState = Arc<RwLock<HashMap<String, NodeRecord>>>;

pub fn new_shared_state() -> SharedState {
    Arc::new(RwLock::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// Shared Runtime State
// ---------------------------------------------------------------------------

/// Fields shared across the main loop and spawned tasks.
pub(crate) struct P2PShared {
    peer_id: String,
    tier: NodeTier,
    peers: SharedState,
    mdns_pending: Arc<RwLock<Vec<(PeerId, libp2p::Multiaddr)>>>,
}

// ---------------------------------------------------------------------------
// P2PRuntime
// ---------------------------------------------------------------------------

const TOPIC_HEARTBEAT: &str = "shear/heartbeat/1";
const TOPIC_CONSENSUS:  &str = "shear/consensus/1";
const TOPIC_VERIFIED:   &str = "shear/verified/1";

pub struct P2PRuntime {
    pub kad_swarm: Swarm<kad::Behaviour<kad::store::MemoryStore>>,
    pub mdns: mdns::tokio::Behaviour,
    /// Wrapped in Arc<Mutex<>> so spawned tasks can access it immutably
    /// while the main loop accesses it via &mut self.
    pub gossipsub: Arc<Mutex<gossipsub::Behaviour>>,
    pub shared: Arc<P2PShared>,
}

impl P2PRuntime {
    pub async fn new(tier: NodeTier, listen_addr: &str) -> anyhow::Result<Self> {
        let keypair = Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(&keypair.public());
        tracing::info!("[P2P] Local PeerId: {}", local_peer_id);

        // Transport: TCP + noise + yamux
        let tcp = libp2p::tcp::tokio::Transport::default();
        let noise = libp2p::noise::Config::new(&keypair)
            .map_err(|e| anyhow::anyhow!("noise: {}", e))?;
        let transport = tcp
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(noise)
            .multiplex(libp2p::yamux::Config::default())
            .boxed();

        // Kademlia
        let kad_behaviour = kad::Behaviour::new(
            local_peer_id,
            kad::store::MemoryStore::new(local_peer_id),
        );

        // mDNS
        let mdns_behaviour = mdns::tokio::Behaviour::new(
            mdns::Config::default(),
            local_peer_id,
        ).map_err(|e| anyhow::anyhow!("mdns: {}", e))?;

        // Gossipsub
        let gs_config = gossipsub::ConfigBuilder::default()
            .build()
            .map_err(|e| anyhow::anyhow!("gossipsub: {}", e))?;
        let gossipsub_behaviour = match gossipsub::Behaviour::new(
            libp2p::gossipsub::MessageAuthenticity::Signed(keypair),
            gs_config,
        ) {
            Ok(b) => b,
            Err(e) => return Err(anyhow::anyhow!("gossipsub: {}", e)),
        };

        // Kad Swarm
        let kad_swarm = Swarm::new(
            transport,
            kad_behaviour,
            local_peer_id,
            libp2p::swarm::Config::with_tokio_executor(),
        );

        tracing::info!("[P2P] {} starting as tier {} on {}", local_peer_id, tier, listen_addr);

        let peer_id = local_peer_id.to_base58();

        Ok(Self {
            kad_swarm,
            mdns: mdns_behaviour,
            gossipsub: Arc::new(Mutex::new(gossipsub_behaviour)),
            shared: Arc::new(P2PShared {
                peer_id,
                tier,
                peers: new_shared_state(),
                mdns_pending: Arc::new(RwLock::new(Vec::new())),
            }),
        })
    }

    pub fn subscribe_topics(&mut self) -> anyhow::Result<()> {
        let mut gs = self.gossipsub.lock().unwrap();
        for name in [TOPIC_HEARTBEAT, TOPIC_CONSENSUS, TOPIC_VERIFIED] {
            let topic = gossipsub::IdentTopic::new(name);
            gs.subscribe(&topic)
                .map_err(|e| anyhow::anyhow!("subscribe {}: {}", name, e))?;
        }
        tracing::info!("[P2P] Subscribed to 3 gossip topics");
        Ok(())
    }

    pub fn dial(&mut self, addr: &str) -> anyhow::Result<()> {
        let addr: libp2p::Multiaddr = addr.parse()?;
        let addr_display = addr.to_string();
        self.kad_swarm.dial(addr)
            .map_err(|e| anyhow::anyhow!("dial: {}", e))?;
        tracing::info!("[P2P] Dialing {}", addr_display);
        Ok(())
    }

    pub fn bootstrap(&mut self, peer_id: &str, addr: &str) -> anyhow::Result<()> {
        let pid: PeerId = peer_id.parse()?;
        let multiaddr: libp2p::Multiaddr = addr.parse()?;
        self.kad_swarm.behaviour_mut().add_address(&pid, multiaddr);
        tracing::info!("[P2P] Bootstrap seed: {} at {}", peer_id, addr);
        Ok(())
    }

    pub async fn publish_record(&mut self) -> anyhow::Result<()> {
        let record = NodeRecord::new(&self.shared.peer_id, self.shared.tier);
        let value = serde_json::to_vec(&record)?;
        self.kad_swarm.behaviour_mut().put_record(
            libp2p::kad::Record::new(
                libp2p::kad::RecordKey::new(b"shear_node_v1"),
                value,
            ),
            kad::Quorum::One,
        )?;
        self.shared.peers.write().await.insert(self.shared.peer_id.clone(), record);
        tracing::info!("[P2P] Published record to Kad DHT");
        Ok(())
    }

    /// Broadcast heartbeat via Gossipsub.
    pub fn send_heartbeat(&self) {
        let hb = Heartbeat::now(&self.shared.peer_id, self.shared.tier);
        if let Ok(payload) = serde_json::to_vec(&hb) {
            let topic = gossipsub::IdentTopic::new(TOPIC_HEARTBEAT);
            if let Ok(mut gs) = self.gossipsub.lock() {
                if let Err(e) = gs.publish(topic, payload) {
                    tracing::warn!("[P2P] Heartbeat failed: {}", e);
                }
            }
        }
    }

    /// Broadcast consensus request.
    pub fn broadcast_request(&self, req: &InferenceRequest) {
        if let Ok(payload) = serde_json::to_vec(req) {
            let topic = gossipsub::IdentTopic::new(TOPIC_CONSENSUS);
            if let Ok(mut gs) = self.gossipsub.lock() {
                if let Err(e) = gs.publish(topic, payload) {
                    tracing::warn!("[P2P] Broadcast req failed: {}", e);
                }
            }
        }
    }

    /// Broadcast verified result.
    pub fn broadcast_result(&self, result: &InferenceResult) {
        if let Ok(payload) = serde_json::to_vec(result) {
            let topic = gossipsub::IdentTopic::new(TOPIC_VERIFIED);
            if let Ok(mut gs) = self.gossipsub.lock() {
                if let Err(e) = gs.publish(topic, payload) {
                    tracing::warn!("[P2P] Broadcast result failed: {}", e);
                }
            }
        }
    }

    pub async fn peers_of_tier(&self, tier: NodeTier) -> Vec<String> {
        let peers = self.shared.peers.read().await;
        peers.values().filter(|r| r.tier == tier && r.available)
            .map(|r| r.peer_id.clone()).collect()
    }

    pub async fn all_peers(&self) -> Vec<String> {
        let peers = self.shared.peers.read().await;
        peers.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Spawned Tasks
// ---------------------------------------------------------------------------

/// mDNS polling task: discovers LAN peers and queues them for Kad.
/// Uses std::thread::spawn because Context is not Send.
fn spawn_mdns_task(
    mdns: mdns::tokio::Behaviour,
    shared: Arc<P2PShared>,
) {
    std::thread::spawn(move || {
        use std::pin::pin;
        use std::task::Context;
        use futures::task::noop_waker_ref;
        use libp2p::swarm::ToSwarm;

        let waker = noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        let mut mdns = pin!(mdns);

        loop {
            match mdns.as_mut().poll(&mut cx) {
                std::task::Poll::Ready(ToSwarm::GenerateEvent(event)) => {
                    if let mdns::Event::Discovered(addrs) = event {
                        for (peer_id, addr) in addrs {
                            tracing::info!("[P2P] mDNS: discovered {} at {}", peer_id, addr);
                            // Use blocking write since we're in a sync context
                            let pending = shared.mdns_pending.clone();
                            let rt = tokio::runtime::Handle::current();
                            rt.block_on(async {
                                pending.write().await.push((peer_id, addr));
                            });
                        }
                    }
                }
                std::task::Poll::Ready(_) => {}  // other ToSwarm variants
                std::task::Poll::Pending => {}
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });
}

// ---------------------------------------------------------------------------
// Main Event Loop
// ---------------------------------------------------------------------------

/// Main P2P event loop.
/// Kad Swarm is the primary stream (implements StreamExt).
/// mDNS is polled in a spawned task and forwards peers to Kad.
/// Gossipsub is accessed via Arc<Mutex<>> for heartbeats and broadcasts.
pub async fn run_p2p(mut runtime: P2PRuntime, app_peers: SharedState) {
    use libp2p::futures::StreamExt;

    // Publish our record on startup
    if let Err(e) = runtime.publish_record().await {
        tracing::warn!("[P2P] Record publish failed: {}", e);
    }

    // Extract mDNS before spawning (avoids partial move of runtime)
    let mdns = std::mem::replace(&mut runtime.mdns, unsafe { std::mem::zeroed() });
    // Spawn mDNS polling task (uses std::thread since Context is not Send)
    spawn_mdns_task(mdns, runtime.shared.clone());

    // Spawn heartbeat task (self-contained, uses Arc<Mutex<gossipsub>>)
    {
        let gs_heartbeat = runtime.gossipsub.clone();
        let shared = runtime.shared.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                let hb = Heartbeat::now(&shared.peer_id, shared.tier);
                if let Ok(payload) = serde_json::to_vec(&hb) {
                    let topic = gossipsub::IdentTopic::new(TOPIC_HEARTBEAT);
                    if let Ok(mut gs) = gs_heartbeat.lock() {
                        let _ = gs.publish(topic, payload);
                    }
                }
            }
        });
    }

    // Kad mDNS flush interval
    let mut process_mdns_interval = tokio::time::interval(std::time::Duration::from_secs(5));

    loop {
        tokio::select! {
            // Kad Swarm events
            event = runtime.kad_swarm.next() => {
                match event {
                    Some(libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. }) => {
                        tracing::info!("[P2P] Kad: connected to {:?}", peer_id);
                    }
                    Some(libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id, .. }) => {
                        tracing::info!("[P2P] Kad: disconnected {:?}", peer_id);
                    }
                    Some(libp2p::swarm::SwarmEvent::Dialing { peer_id, .. }) => {
                        tracing::debug!("[P2P] Kad: dialing {:?}", peer_id);
                    }
                    Some(libp2p::swarm::SwarmEvent::NewListenAddr { .. }) => {
                        // Kad manages listen addr internally
                    }
                    None => {
                        tracing::warn!("[P2P] Kad Swarm stream ended unexpectedly");
                    }
                    _ => {}
                }
            }

            // Process pending mDNS discoveries → add to Kad
            _ = process_mdns_interval.tick() => {
                let pending: Vec<_> = {
                    let mut q = runtime.shared.mdns_pending.write().await;
                    std::mem::take(&mut *q)
                };
                for (peer_id, addr) in pending {
                    runtime.kad_swarm.behaviour_mut().add_address(&peer_id, addr.clone());
                    tracing::info!("[P2P] Kad: added mDNS peer {} at {}", peer_id, addr);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unix_ts() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}

// ============================================================================
// Backward-compat shim (main.rs)
// ============================================================================

pub const P2P_PORT: u16 = 9090;

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
    pub fn new_local(node_id: String, role: String, http_port: u16, p2p_port: u16) -> Self {
        Self { node_id, role, http_port, p2p_port, available: true, reputation: 1.0 }
    }
}

pub struct P2PState {
    pub local_node: Arc<NetworkNode>,
    pub peers: SharedState,
}

impl P2PState {
    pub fn new(local_node: NetworkNode) -> Self {
        Self { local_node: Arc::new(local_node), peers: new_shared_state() }
    }
}

pub async fn forward_to_worker(
    model: &str, prompt: &str, max_tokens: usize, temperature: f32,
) -> String {
    tracing::debug!("[P2P] forward_to_worker model={} prompt_len={} max_tokens={} temp={}",
        model, prompt.len(), max_tokens, temperature);
    format!("[stub inference for model '{}', {} chars, {} tokens, temp {}]",
        model, prompt.len(), max_tokens, temperature)
}

pub async fn start_p2p_server(peers: SharedState, tier: NodeTier, port: u16) {
    let listen_addr = format!("/ip4/0.0.0.0/tcp/{}", port);
    match P2PRuntime::new(tier, &listen_addr).await {
        Ok(mut runtime) => {
            if let Err(e) = runtime.subscribe_topics() {
                tracing::error!("[P2P] Topic subscription failed: {}", e);
                return;
            }
            tracing::info!("[P2P] Server started on {}", listen_addr);
            run_p2p(runtime, peers).await;
        }
        Err(e) => tracing::error!("[P2P] Server failed: {}", e),
    }
}

pub async fn bootstrap_peers(_peers: SharedState) {
    tracing::info!("[P2P] bootstrap_peers: no seeds configured");
}

pub async fn p2p_peers_handler(
    State(state): State<Arc<crate::AppState>>,
) -> Json<serde_json::Value> {
    let local = &state.p2p_state.local_node;
    let peer_count = state.p2p_state.peers.blocking_read().len();
    Json(serde_json::json!({
        "node_id": local.node_id,
        "role": local.role,
        "p2p_port": local.p2p_port,
        "available": local.available,
        "reputation": local.reputation,
        "credits": state.credits,
        "reputation_app": state.reputation,
        "peer_count": peer_count,
        "peers": [],
    }))
}

pub async fn p2p_connect_handler(
    State(_state): State<Arc<crate::AppState>>,
    axum::extract::Path(addr): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    tracing::info!("[P2P] connect request to {}", addr);
    Json(serde_json::json!({ "status": "ok", "message": "P2P connection requested", "addr": addr }))
}
