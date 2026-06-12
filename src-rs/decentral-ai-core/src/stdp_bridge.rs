//! STDP Bridge: Load agent_demo STDP weights → Modulate RWKV decay/mix parameters
//!
//! SALAMI Architecture:
//!   agent_demo (Brain learner)  →  stdp_weights.json  →  SHEAR (RWKV modulator)
//!
//! Mapping strategy:
//!   - Fact weight ↑ → decay ↓ (longer memory retention, rely on stored facts)
//!   - Hop weight ↑ → mix ↑ (more context fusion for multi-hop reasoning)
//!   - Latent weight ↑ → hidden state dimensions bias
//!   - Curiosity weight ↑ → temperature ↑ (more exploration, sampling diversity)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// Reasoner types from agent_demo (mirror for JSON parsing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReasonerType {
    Fact,
    Hop,
    Latent,
    Curiosity,
    Learn,
}

impl std::str::FromStr for ReasonerType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fact" => Ok(ReasonerType::Fact),
            "hop" => Ok(ReasonerType::Hop),
            "latent" => Ok(ReasonerType::Latent),
            "curiosity" => Ok(ReasonerType::Curiosity),
            "learn" => Ok(ReasonerType::Learn),
            _ => Err(format!("Unknown ReasonerType: {}", s)),
        }
    }
}

/// Loaded STDP weight entry from agent_demo export
#[derive(Debug, Clone, Deserialize)]
pub struct StdpWeightEntry {
    weight: f32,
    success_count: u32,
    fail_count: u32,
    success_rate: f32,
}

/// Modulation parameters for RWKV
#[derive(Debug, Clone)]
pub struct RwkvModulation {
    /// Decay multiplier (applied to att.decay, lower = longer retention)
    pub decay_mult: f32,
    /// Mix multiplier (applied to att.mix_k/r/v, higher = more context fusion)
    pub mix_mult: f32,
    /// Temperature offset for sampling (curiosity ↑ = temp ↑)
    pub temp_offset: f32,
    /// Hidden dimension importance weights (latent path focus)
    pub dim_bias: Vec<f32>,
}

/// STDP Bridge: loads weights from agent_demo and computes RWKV modulation
pub struct StdpBridge {
    weights: HashMap<ReasonerType, StdpWeightEntry>,
    /// Current RWKV hidden dimension (for dim_bias)
    hidden_size: usize,
}

impl StdpBridge {
    /// Load from agent_demo exported JSON
    pub fn load(path: &str, hidden_size: usize) -> Self {
        if !std::path::Path::new(path).exists() {
            return Self::default_weights(hidden_size);
        }
        
        let json = fs::read_to_string(path).expect("Failed to read STDP weights file");
        let weights_json: HashMap<String, StdpWeightEntry> = 
            serde_json::from_str(&json).expect("Failed to parse STDP weights JSON");
        
        let mut weights = HashMap::new();
        for (key, val) in weights_json {
            if let Ok(rt) = key.parse::<ReasonerType>() {
                weights.insert(rt, val);
            }
        }
        
        StdpBridge { weights, hidden_size }
    }
    
    /// Default weights (no modulation)
    fn default_weights(hidden_size: usize) -> Self {
        let mut weights = HashMap::new();
        weights.insert(ReasonerType::Fact, StdpWeightEntry { weight: 0.5, success_count: 0, fail_count: 0, success_rate: 0.5 });
        weights.insert(ReasonerType::Hop, StdpWeightEntry { weight: 0.5, success_count: 0, fail_count: 0, success_rate: 0.5 });
        weights.insert(ReasonerType::Latent, StdpWeightEntry { weight: 0.5, success_count: 0, fail_count: 0, success_rate: 0.5 });
        weights.insert(ReasonerType::Curiosity, StdpWeightEntry { weight: 0.5, success_count: 0, fail_count: 0, success_rate: 0.5 });
        weights.insert(ReasonerType::Learn, StdpWeightEntry { weight: 0.5, success_count: 0, fail_count: 0, success_rate: 0.5 });
        StdpBridge { weights, hidden_size }
    }
    
    /// Compute RWKV modulation from current STDP weights
    pub fn compute_modulation(&self) -> RwkvModulation {
        let fact_w = self.weights.get(&ReasonerType::Fact).map(|w| w.weight).unwrap_or(0.5);
        let hop_w = self.weights.get(&ReasonerType::Hop).map(|w| w.weight).unwrap_or(0.5);
        let latent_w = self.weights.get(&ReasonerType::Latent).map(|w| w.weight).unwrap_or(0.5);
        let curiosity_w = self.weights.get(&ReasonerType::Curiosity).map(|w| w.weight).unwrap_or(0.5);
        
        // Mapping formulas (tuned for RWKV-4-430M)
        // Fact ↑ → decay ↓ (keep memory longer)
        //   decay_mult = 1.0 - 0.3 * (fact_w - 0.5)  // range [0.85, 1.15]
        let decay_mult = 1.0 - 0.3 * (fact_w - 0.5);
        
        // Hop ↑ → mix ↑ (blend context more for multi-hop)
        //   mix_mult = 0.8 + 0.4 * hop_w  // range [0.8, 1.2]
        let mix_mult = 0.8 + 0.4 * hop_w;
        
        // Curiosity ↑ → temp ↑ (explore more)
        //   temp_offset = 0.5 * (curiosity_w - 0.5)  // range [-0.25, 0.25]
        let temp_offset = 0.5 * (curiosity_w - 0.5);
        
        // Latent weight → hidden dimension bias
        //   Top 32 dims get boosted by latent_w * 0.1
        let dim_bias = vec![latent_w * 0.1; self.hidden_size];
        
        RwkvModulation { decay_mult, mix_mult, temp_offset, dim_bias }
    }
    
    /// Get current weight for a reasoner type
    pub fn get_weight(&self, rt: ReasonerType) -> f32 {
        self.weights.get(&rt).map(|w| w.weight).unwrap_or(0.5)
    }
    
    /// Describe current state
    pub fn describe(&self) -> String {
        let m = self.compute_modulation();
        let mut lines = vec!["🌉 STDP Bridge State:".to_string()];
        lines.push(format!("  Fact→Decay: {:.2} (decay_mult={:.2})", self.get_weight(ReasonerType::Fact), m.decay_mult));
        lines.push(format!("  Hop→Mix: {:.2} (mix_mult={:.2})", self.get_weight(ReasonerType::Hop), m.mix_mult));
        lines.push(format!("  Latent→DimBias: {:.2}", self.get_weight(ReasonerType::Latent)));
        lines.push(format!("  Curiosity→Temp: {:.2} (temp_offset={:.2})", self.get_weight(ReasonerType::Curiosity), m.temp_offset));
        lines.join("\n")
    }
}

impl Default for StdpBridge {
    fn default() -> Self {
        Self::default_weights(1024) // RWKV-4-430M hidden size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_bridge() {
        let bridge = StdpBridge::default();
        let m = bridge.compute_modulation();
        // Default weights = 0.5 → neutral modulation
        assert!((m.decay_mult - 1.0).abs() < 0.01);
        assert!((m.mix_mult - 1.0).abs() < 0.01);
        assert!((m.temp_offset).abs() < 0.01);
    }
    
    #[test]
    fn test_fact_high_decay_low() {
        let mut bridge = StdpBridge::default_weights(1024);
        bridge.weights.get_mut(&ReasonerType::Fact).unwrap().weight = 0.8;
        let m = bridge.compute_modulation();
        // Fact ↑ 0.8 → decay_mult = 1.0 - 0.3*(0.8-0.5) = 0.91
        assert!((m.decay_mult - 0.91).abs() < 0.01);
    }
    
    #[test]
    fn test_hop_high_mix_high() {
        let mut bridge = StdpBridge::default_weights(1024);
        bridge.weights.get_mut(&ReasonerType::Hop).unwrap().weight = 0.9;
        let m = bridge.compute_modulation();
        // Hop ↑ 0.9 → mix_mult = 0.8 + 0.4*0.9 = 1.16
        assert!((m.mix_mult - 1.16).abs() < 0.01);
    }
    
    #[test]
    fn test_curiosity_high_temp_high() {
        let mut bridge = StdpBridge::default_weights(1024);
        bridge.weights.get_mut(&ReasonerType::Curiosity).unwrap().weight = 0.7;
        let m = bridge.compute_modulation();
        // Curiosity ↑ 0.7 → temp_offset = 0.5*(0.7-0.5) = 0.1
        assert!((m.temp_offset - 0.1).abs() < 0.01);
    }
    
    #[test]
    fn test_load_missing_file() {
        let bridge = StdpBridge::load("/nonexistent/path.json", 1024);
        // Should return default weights
        assert!((bridge.get_weight(ReasonerType::Fact) - 0.5).abs() < 0.01);
    }
}