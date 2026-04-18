// SHEAR Core — Library root

pub mod aggregator;
pub mod cell;
pub mod network;
pub mod router;
pub mod rwkv_model;
pub mod rwkv_weights;
pub mod tokenizer;

use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct AppState {
    pub node_id: String,
    pub role: NodeRole,
    pub credits: f64,
    pub reputation: f64,
    pub p2p_state: network::SharedState,
}

impl AppState {
    pub fn new(node_id: String, role: NodeRole, p2p_state: network::SharedState) -> Self {
        Self { node_id, role, credits: 100.0, reputation: 50.0, p2p_state }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeRole {
    L0Collector, L1Lightweight, L2Standard, L3Heavy, L4Datacenter,
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeRole::L0Collector => write!(f, "L0"),
            NodeRole::L1Lightweight => write!(f, "L1"),
            NodeRole::L2Standard => write!(f, "L2"),
            NodeRole::L3Heavy => write!(f, "L3"),
            NodeRole::L4Datacenter => write!(f, "L4"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage { pub role: String, pub content: String }
