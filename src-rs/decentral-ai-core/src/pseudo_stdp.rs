//! Pseudo-STDP: lightweight attention-like mechanism on top of RWKV
//!
//! Core idea: Track which hidden dimensions are most "active" for each token,
//! and use this to bias sampling toward tokens that activate similar patterns.

use crate::rwkv_model::{RwkvModel, RwkvModelState, VOCAB};

/// Pseudo-STDP state: tracks "active" dimensions per recent context
#[derive(Clone)]
pub struct PseudoStdpState {
    recent_activations: Vec<(usize, f32)>,
    window_size: usize,
    decay: f32,
}

impl PseudoStdpState {
    pub fn new(window_size: usize, decay: f32) -> Self {
        Self { recent_activations: Vec::new(), window_size, decay }
    }

    /// Record activation pattern from one token's forward pass
    pub fn record(&mut self, hidden: &[f32]) {
        let k = 32.min(hidden.len());
        let mut indexed: Vec<(usize, f32)> = hidden.iter().enumerate()
            .map(|(i, v)| (i, *v)).collect();
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        
        for (dim, act) in indexed.into_iter().take(k) {
            for (_, existing_act) in &mut self.recent_activations {
                *existing_act *= self.decay;
            }
            if let Some(existing) = self.recent_activations.iter_mut().find(|(d, _)| *d == dim) {
                existing.1 += act;
            } else {
                self.recent_activations.push((dim, act));
            }
        }
        if self.recent_activations.len() > self.window_size * 2 {
            self.recent_activations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
            self.recent_activations.truncate(self.window_size);
        }
    }

    /// Get bias for next token based on activation pattern
    pub fn get_bias(&self, model: &RwkvModel) -> Vec<f32> {
        let mut bias = vec![0.0f32; VOCAB];
        if self.recent_activations.is_empty() { return bias; }
        
        let avg_importance: f32 = self.recent_activations.iter().map(|(_, a)| a.abs()).sum::<f32>() 
            / self.recent_activations.len().max(1) as f32;
        
        for (dim, weight) in &self.recent_activations {
            if weight.abs() < avg_importance * 0.1 { continue; }
            let sample_step = (VOCAB / 256).max(1);
            for token_id in (0..VOCAB).step_by(sample_step) {
                let emb_val = model.emb.row(token_id).into_iter().nth(*dim).unwrap_or(&0.0);
                bias[token_id] += weight * emb_val * 0.01;
            }
        }
        bias
    }
}

/// Generate with pseudo-STDP bias
pub fn generate_with_stdp(
    model: &RwkvModel, input_ids: &[usize], max_new: usize, temperature: f32,
    stdp_window: usize, stdp_decay: f32,
) -> (Vec<usize>, std::time::Duration, PseudoStdpState) {
    use std::time::Instant;
    let start = Instant::now();
    let mut state = RwkvModelState::new();
    let stdp_state = PseudoStdpState::new(stdp_window, stdp_decay);
    let mut all_ids = input_ids.to_vec();

    for &id in input_ids { model.forward_token(id, &mut state); }

    for _ in 0..max_new {
        let logits = model.forward_token(*all_ids.last().unwrap(), &mut state);
        let bias = stdp_state.get_bias(model);
        let biased_logits: Vec<f32> = logits.iter().zip(bias.iter()).map(|(l, b)| l + b).collect();
        let next = sample_token_stdp(&biased_logits, temperature);
        all_ids.push(next);
        if next == 0 { break; }
    }
    (all_ids, start.elapsed(), stdp_state)
}

fn sample_token_stdp(logits: &[f32], temperature: f32) -> usize {
    use rand::Rng;
    if temperature < 1e-6 {
        return logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
    }
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
    let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_s: Vec<f32> = scaled.iter().map(|v| (v - max_s).exp()).collect();
    let sum_exp = exp_s.iter().sum::<f32>();
    let probs: Vec<f32> = exp_s.iter().map(|v| v / sum_exp).collect();
    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = 0.9_f32;
    let mut cumsum = 0.0_f32;
    let mut top: Vec<(usize, f32)> = Vec::new();
    for p in &pairs { if cumsum >= threshold { break; } cumsum += p.1; top.push(*p); }
    let total: f32 = top.iter().map(|(_, p)| p).sum();
    let r: f32 = rand::thread_rng().gen();
    let mut acc = 0.0_f32;
    for (i, p) in top { acc += p / total; if r <= acc { return i; } }
    pairs.last().map(|(i, _)| *i).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_state_creation() {
        let state = PseudoStdpState::new(10, 0.9);
        assert_eq!(state.window_size, 10);
        assert_eq!(state.decay, 0.9);
    }

    #[test]
    fn test_stdp_record() {
        let mut state = PseudoStdpState::new(5, 0.9);
        let hidden = vec![0.1; 1024];
        state.record(&hidden);
        assert!(!state.recent_activations.is_empty());
    }
}