// DecentralAI - Model inference module
// Handles model loading and inference for various model types

use serde::{Deserialize, Serialize};

/// Supported model architectures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelArchitecture {
    RWKV4,
    RWKV5,
    RWKV6,
    Llama2,
    Qwen,
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub params: u32,        // Number of parameters (in millions)
    pub context_length: usize,
    pub quantization: String,
}

impl ModelConfig {
    /// RWKV-4-169M configuration
    pub fn rwkv4_169m() -> Self {
        Self {
            architecture: ModelArchitecture::RWKV4,
            params: 169,
            context_length: 4096,
            quantization: "fp32".to_string(),
        }
    }
    
    /// RWKV-4-430M configuration
    pub fn rwkv4_430m() -> Self {
        Self {
            architecture: ModelArchitecture::RWKV4,
            params: 430,
            context_length: 4096,
            quantization: "fp16".to_string(),
        }
    }
}

/// Inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub text: String,
    pub tokens: usize,
    pub latency_ms: u64,
    pub model: String,
}

/// Node capabilities based on hardware
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeTier {
    L0Collector,     // CPU only, data collection
    L1Lightweight,  // CPU + 4GB RAM, 0.5-1.5B models
    L2Standard,     // GPU (RTX 3060), 7B models
    L3Heavy,        // GPU (RTX 3090/4090), 14B+ models
    L4Datacenter,   // A100/H100, 70B backbone
}

impl NodeTier {
    /// Get max model size for this tier (in millions of parameters)
    pub fn max_params(&self) -> u32 {
        match self {
            NodeTier::L0Collector => 0,
            NodeTier::L1Lightweight => 1500,
            NodeTier::L2Standard => 7000,
            NodeTier::L3Heavy => 14000,
            NodeTier::L4Datacenter => 70000,
        }
    }
    
    /// Check if this tier can run a given model
    pub fn can_run(&self, config: &ModelConfig) -> bool {
        config.params <= self.max_params()
    }
}

/// Router decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterDecision {
    pub selected_node: String,
    pub tier: NodeTier,
    pub estimated_latency_ms: u64,
    pub cost_credits: f64,
}

impl RouterDecision {
    /// Simple routing based on model size
    pub fn route_for_model(config: &ModelConfig) -> Self {
        let tier = match config.params {
            0..=500 => NodeTier::L1Lightweight,
            501..=3000 => NodeTier::L2Standard,
            3001..=10000 => NodeTier::L3Heavy,
            _ => NodeTier::L4Datacenter,
        };
        
        // Estimate cost (credits per 1K tokens)
        let base_cost = match tier {
            NodeTier::L0Collector => 0.1,
            NodeTier::L1Lightweight => 0.5,
            NodeTier::L2Standard => 1.5,
            NodeTier::L3Heavy => 3.0,
            NodeTier::L4Datacenter => 8.0,
        };
        
        Self {
            selected_node: format!("{:?}Node", tier),
            tier,
            estimated_latency_ms: 1000, // Placeholder
            cost_credits: base_cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_tier_limits() {
        assert!(NodeTier::L1Lightweight.can_run(&ModelConfig::rwkv4_169m()));
        assert!(!NodeTier::L0Collector.can_run(&ModelConfig::rwkv4_430m()));
    }
    
    #[test]
    fn test_router_decision() {
        let config = ModelConfig::rwkv4_169m();
        let decision = RouterDecision::route_for_model(&config);
        assert_eq!(decision.tier, NodeTier::L1Lightweight);
    }
}