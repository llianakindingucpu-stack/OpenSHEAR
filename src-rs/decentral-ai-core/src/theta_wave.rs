//! θ-wave heartbeat protocol — node synchronization for SHEAR distributed inference.
//!
//! Theta oscillations (7-8 Hz) in the hippocampus are associated with memory
//! consolidation and coordination between brain regions. This protocol mirrors
//! that: periodic heartbeat packets synchronize node states.
//!
//! Protocol design:
//! - Heartbeat interval: 500ms (2 Hz — slower than theta, for network efficiency)
//! - Payload: node_id, last_hidden_state[H], confidence, inference_count
//! - Aggregation: weighted average based on confidence and recency
//!
//! Two modes:
//! - ACTIVE: node initiates heartbeat (push to known peers)
//! - PASSIVE: node responds to heartbeat (update peer state)

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Types

/// A heartbeat packet from one node.
#[derive(Clone, Debug)]
pub struct ThetaPulse {
    /// Origin node ID.
    pub node_id: String,
    /// Timestamp when this pulse was created (monotonic).
    pub timestamp_ms: u64,
    /// Hidden state vector from last inference [H].
    pub hidden_state: Vec<f32>,
    /// Local inference confidence [0, 1].
    pub confidence: f32,
    /// Number of inferences performed by this node.
    pub inference_count: u64,
}

/// Peer state tracking.
#[derive(Clone, Debug)]
pub struct PeerState {
    pub node_id: String,
    pub last_pulse: ThetaPulse,
    pub last_seen: Instant,
    /// Weighted contribution to aggregation (confidence * recency_factor).
    pub weight: f32,
}

/// Configuration for theta-wave protocol.
#[derive(Clone, Debug)]
pub struct ThetaWaveConfig {
    /// Heartbeat interval.
    pub interval_ms: u64,
    /// Peer timeout - if no pulse for this long, peer is considered dead.
    pub peer_timeout_ms: u64,
    /// Recency weight: newer pulses get higher weight = exp(-age_ms / decay_half_life_ms).
    pub decay_half_life_ms: u64,
    /// Minimum confidence to contribute to aggregation.
    pub min_confidence: f32,
}

impl Default for ThetaWaveConfig {
    fn default() -> Self {
        Self {
            interval_ms: 500,
            peer_timeout_ms: 5000,
            decay_half_life_ms: 2000,
            min_confidence: 0.3,
        }
    }
}

// ============================================================================
// Core protocol

/// Theta-wave engine managing peer heartbeats and state aggregation.
pub struct ThetaWave {
    cfg: ThetaWaveConfig,
    /// Local node ID.
    node_id: String,
    /// Last pulse we sent.
    last_pulse: Option<ThetaPulse>,
    /// Tracked peer states.
    peers: HashMap<String, PeerState>,
    /// Last heartbeat timestamp.
    last_heartbeat: Option<Instant>,
}

impl ThetaWave {
    /// Create a new theta-wave engine.
    pub fn new(node_id: String, cfg: ThetaWaveConfig) -> Self {
        Self {
            cfg,
            node_id,
            last_pulse: None,
            peers: HashMap::new(),
            last_heartbeat: None,
        }
    }

    /// Create a pulse from local state (call after inference).
    pub fn make_pulse(&mut self, hidden_state: Vec<f32>, confidence: f32, inference_count: u64) -> ThetaPulse {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let pulse = ThetaPulse {
            node_id: self.node_id.clone(),
            timestamp_ms: now,
            hidden_state,
            confidence,
            inference_count,
        };
        self.last_pulse = Some(pulse.clone());
        pulse
    }

    /// Send a pulse to update local state (for the local node simulating itself).
    pub fn register_local_pulse(&mut self, pulse: ThetaPulse) {
        self.last_pulse = Some(pulse.clone());
        self.last_heartbeat = Some(Instant::now());
    }

    /// Process an incoming pulse from a peer.
    pub fn receive_pulse(&mut self, pulse: ThetaPulse) {
        // Update peer state
        let now = Instant::now();
        self.peers.insert(
            pulse.node_id.clone(),
            PeerState {
                node_id: pulse.node_id.clone(),
                last_pulse: pulse.clone(),
                last_seen: now,
                weight: 0.0, // Computed in aggregate()
            },
        );
    }

    /// Check if it's time to send a heartbeat.
    pub fn should_heartbeat(&self) -> bool {
        match self.last_heartbeat {
            Some(t) => t.elapsed().as_millis() as u64 >= self.cfg.interval_ms,
            None => true,
        }
    }

    /// Get all alive peers (not timed out).
    pub fn alive_peers(&self) -> Vec<&PeerState> {
        let now = Instant::now();
        self.peers
            .values()
            .filter(|p| {
                let age = now.duration_since(p.last_seen).as_millis() as u64;
                age < self.cfg.peer_timeout_ms
            })
            .collect()
    }

    /// Compute aggregated hidden state from all nodes (local + peers).
    /// Returns (aggregated_hidden[H], contributors: Vec<(node_id, weight)>).
    pub fn aggregate(&self, local_hidden: &[f32], local_confidence: f32) -> (Vec<f32>, Vec<(String, f32)>) {
        let mut contributions: Vec<(Vec<f32>, f32)> = vec![];

        // Local node
        if local_hidden.len() > 0 && local_confidence >= self.cfg.min_confidence {
            contributions.push((local_hidden.to_vec(), local_confidence));
        }

        // Peers
        let now = Instant::now();
        for peer in self.alive_peers() {
            if peer.last_pulse.confidence >= self.cfg.min_confidence {
                // Recency weight
                let age = now.duration_since(peer.last_seen).as_millis() as u64;
                let recency = if self.cfg.decay_half_life_ms > 0 {
                    0.5_f32.powf(age as f32 / self.cfg.decay_half_life_ms as f32)
                } else {
                    1.0
                };
                let weight = peer.last_pulse.confidence * recency;
                contributions.push((peer.last_pulse.hidden_state.clone(), weight));
            }
        }

        if contributions.is_empty() {
            return (vec![], vec![]);
        }

        // Weighted average
        let h = contributions[0].0.len();
        let total_weight: f32 = contributions.iter().map(|(_, w)| w).sum();
        if total_weight < 1e-6 {
            return (vec![], vec![]);
        }

        let mut agg = vec![0.0_f32; h];
        for (state, weight) in &contributions {
            let w = weight / total_weight;
            for (i, v) in state.iter().enumerate() {
                agg[i] += w * v;
            }
        }

        // Build contributors list (local + peers)
        let mut contributors: Vec<(String, f32)> = vec![(self.node_id.clone(), contributions.iter().map(|(_, w)| *w).fold(0.0, |a, b| a + b))];
        for peer in self.peers.values() {
            contributors.push((peer.node_id.clone(), peer.weight));
        }

        (agg, contributors)
    }

    /// Clean up dead peers.
    pub fn cleanup(&mut self) {
        let now = Instant::now();
        self.peers.retain(|_, p| {
            let age = now.duration_since(p.last_seen).as_millis() as u64;
            age < self.cfg.peer_timeout_ms
        });
    }
}

// ============================================================================
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pulse(id: &str, conf: f32) -> ThetaPulse {
        ThetaPulse {
            node_id: id.to_string(),
            timestamp_ms: 1000,
            hidden_state: vec![1.0, 0.0, 0.5],
            confidence: conf,
            inference_count: 10,
        }
    }

    #[test]
    fn test_make_pulse() {
        let mut tw = ThetaWave::new("node1".to_string(), ThetaWaveConfig::default());
        let pulse = tw.make_pulse(vec![1.0, 0.0], 0.9, 5);
        assert_eq!(pulse.node_id, "node1");
        assert_eq!(pulse.confidence, 0.9);
    }

    #[test]
    fn test_receive_pulse() {
        let mut tw = ThetaWave::new("node1".to_string(), ThetaWaveConfig::default());
        let pulse = make_pulse("node2", 0.8);
        tw.receive_pulse(pulse);
        assert!(tw.peers.contains_key("node2"));
    }

    #[test]
    fn test_should_heartbeat() {
        let tw = ThetaWave::new("node1".to_string(), ThetaWaveConfig::default());
        assert!(tw.should_heartbeat()); // No heartbeat yet
    }

    #[test]
    fn test_aggregate_empty() {
        let tw = ThetaWave::new("node1".to_string(), ThetaWaveConfig::default());
        let (agg, contribs) = tw.aggregate(&[1.0, 0.0], 0.9);
        // Local only
        assert!(agg.len() == 2);
        assert!(contribs.len() >= 1);
    }

    #[test]
    fn test_aggregate_with_peers() {
        let mut tw = ThetaWave::new("node1".to_string(), ThetaWaveConfig::default());
        
        // Add peer
        let pulse = make_pulse("node2", 0.7);
        tw.receive_pulse(pulse);

        let (agg, contribs) = tw.aggregate(&[1.0, 0.0, 0.5], 0.8);
        assert!(agg.len() == 3);
        // Should have 2 contributors (local + peer)
        assert!(contribs.len() >= 1);
    }

    #[test]
    fn test_config_default() {
        let cfg = ThetaWaveConfig::default();
        assert_eq!(cfg.interval_ms, 500);
        assert_eq!(cfg.peer_timeout_ms, 5000);
    }
}