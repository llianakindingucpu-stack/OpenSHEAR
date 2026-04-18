//! SHEAR Speculative Decoding Engine
//!
//! Architecture: N Cells (shared weights, independent states) → ensemble voting → output
//!
//! Phase 1 (current): Single-machine, single-model, multi-state ensemble voting
//!   - All cells share the same RwkvModel weights (read-only)
//!   - Each cell has its own RwkvModelState + temperature
//!   - Diversity comes from different sampling temperatures
//!   - Per-token voting: all cells produce logits, vote on next token
//!
//! Phase 2 (future): Cross-node speculative decoding with k-token drafting
//!   - L0/L1 nodes as draft cells (small model, fast)
//!   - L2/L3 nodes as verifiers (large model, accurate)
//!   - Draft k tokens → verify → accept/reject with rollback

use crate::rwkv_model::{RwkvModel, RwkvModelState};
use std::collections::HashMap;
use std::time::Instant;
use rayon::prelude::*;

// ===================== Config =====================

#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of parallel draft cells
    pub n_cells: usize,
    /// Per-cell sampling temperatures (source of diversity)
    pub temperatures: Vec<f32>,
    /// Voting strategy
    pub strategy: VoteStrategy,
    /// Minimum consensus ratio for acceptance (0.0-1.0)
    pub min_consensus: f32,
    /// Top-p sampling threshold
    pub top_p: f32,
    /// Number of tokens to draft per speculative round (for k-token mode)
    pub draft_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VoteStrategy {
    /// Majority vote: most common token wins
    Majority,
    /// Confidence-weighted: weight votes by cell's confidence (max probability)
    ConfidenceWeighted,
    /// First-cell-wins: cell 0 (lowest temperature) decides, others confirm
    LeaderFollow,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            n_cells: 3,
            temperatures: vec![0.5, 0.8, 1.2],
            strategy: VoteStrategy::Majority,
            min_consensus: 0.5,
            top_p: 0.9,
            draft_tokens: 4,
        }
    }
}

impl SpeculativeConfig {
    pub fn with_cells(n: usize) -> Self {
        let temps: Vec<f32> = (0..n).map(|i| {
            match i {
                0 => 0.5,
                1 => 0.8,
                2 => 1.0,
                3 => 1.2,
                _ => 1.0 + 0.2 * (i as f32 - 3.0),
            }
        }).collect();
        Self {
            n_cells: n,
            temperatures: temps,
            strategy: VoteStrategy::Majority,
            min_consensus: 0.5,
            top_p: 0.9,
            draft_tokens: 4,
        }
    }
}

// ===================== Stats =====================

#[derive(Debug, Clone, Default)]
pub struct SpecStats {
    pub total_tokens: usize,
    pub rounds: usize,
    pub consensus_levels: Vec<f32>,
    pub cell_hits: Vec<usize>,
    pub total_time_ms: u64,
    /// For speculative mode: accepted draft tokens
    pub accepted_draft: usize,
    pub rejected_draft: usize,
}

impl SpecStats {
    pub fn avg_consensus(&self) -> f32 {
        if self.consensus_levels.is_empty() { return 0.0; }
        self.consensus_levels.iter().sum::<f32>() / self.consensus_levels.len() as f32
    }

    pub fn tokens_per_second(&self) -> f32 {
        if self.total_time_ms == 0 { return 0.0; }
        self.total_tokens as f32 / (self.total_time_ms as f32 / 1000.0)
    }

    pub fn cell_hit_rates(&self) -> Vec<f32> {
        let total = self.total_tokens as f32;
        if total == 0.0 { return vec![0.0; self.cell_hits.len()]; }
        self.cell_hits.iter().map(|&h| h as f32 / total).collect()
    }

    pub fn acceptance_rate(&self) -> f32 {
        let total = self.accepted_draft + self.rejected_draft;
        if total == 0 { return 0.0; }
        self.accepted_draft as f32 / total as f32
    }
}

// ===================== Engine =====================

pub struct SpeculativeEngine {
    pub model: RwkvModel,
    pub config: SpeculativeConfig,
    pub states: Vec<RwkvModelState>,
    /// Last logits for each cell (from forward_token of the last processed token)
    pending_logits: Vec<Vec<f32>>,
    stats: SpecStats,
}

impl SpeculativeEngine {
    pub fn new(model: RwkvModel, config: SpeculativeConfig) -> Self {
        let n = config.n_cells;
        let states = (0..n).map(|_| RwkvModelState::new()).collect();
        let pending_logits = vec![vec![]; n];
        let cell_hits = vec![0; n];

        Self {
            model,
            config,
            states,
            pending_logits,
            stats: SpecStats { cell_hits, ..Default::default() },
        }
    }

    /// Process prompt tokens through all cells (synchronize states) - PARALLEL
    pub fn process_prompt(&mut self, tokens: &[usize]) {
        let states = std::mem::take(&mut self.states);
        let model = &self.model;

        let (new_states, new_logits): (Vec<_>, Vec<_>) = states
            .into_par_iter()
            .map(|mut state| {
                state.reset();
                let mut last_logits = Vec::new();
                for &id in tokens {
                    last_logits = model.forward_token(id, &mut state);
                }
                (state, last_logits)
            })
            .unzip();

        self.states = new_states;
        self.pending_logits = new_logits;
    }

    /// Sample a token from logits with temperature and top-p
    fn sample_from_logits(&self, logits: &[f32], cell_idx: usize) -> usize {
        let temp = self.config.temperatures.get(cell_idx).copied().unwrap_or(0.8);
        sample_top_p(logits, temp, self.config.top_p)
    }

    /// Generate one token via ensemble voting
    /// Returns (voted_token, consensus_ratio)
    pub fn generate_token(&mut self) -> (usize, f32) {
        let n = self.config.n_cells;

        // 1. Each cell samples from its pending logits
        let cell_tokens: Vec<usize> = (0..n)
            .map(|i| self.sample_from_logits(&self.pending_logits[i], i))
            .collect();

        // 2. Vote on the token
        let (voted, consensus) = self.vote(&cell_tokens);

        // 3. Feed the voted token to ALL cells (synchronize) - PARALLEL
        let states = std::mem::take(&mut self.states);
        let model = &self.model;
        let voted_token = voted;

        let (new_states, new_logits): (Vec<_>, Vec<_>) = states
            .into_par_iter()
            .map(|mut state| {
                let logits = model.forward_token(voted_token, &mut state);
                (state, logits)
            })
            .unzip();

        self.states = new_states;
        self.pending_logits = new_logits;

        // 4. Track which cell "won" (suggested the voted token)
        for (i, &t) in cell_tokens.iter().enumerate() {
            if t == voted {
                self.stats.cell_hits[i] += 1;
            }
        }

        // 5. Update stats
        self.stats.total_tokens += 1;
        self.stats.rounds += 1;
        self.stats.consensus_levels.push(consensus);

        (voted, consensus)
    }

    /// Vote on tokens from N cells
    fn vote(&self, cell_tokens: &[usize]) -> (usize, f32) {
        match self.config.strategy {
            VoteStrategy::Majority => self.vote_majority(cell_tokens),
            VoteStrategy::ConfidenceWeighted => self.vote_confidence_weighted(cell_tokens),
            VoteStrategy::LeaderFollow => self.vote_leader_follow(cell_tokens),
        }
    }

    /// Majority vote: most common token wins
    fn vote_majority(&self, cell_tokens: &[usize]) -> (usize, f32) {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &t in cell_tokens {
            *counts.entry(t).or_insert(0) += 1;
        }

        let (&winner, &count) = counts.iter()
            .max_by_key(|(_, &c)| c)
            .unwrap_or((&0, &0));

        let consensus = count as f32 / cell_tokens.len() as f32;
        (winner, consensus)
    }

    /// Confidence-weighted vote: weight each cell's choice by its confidence
    fn vote_confidence_weighted(&self, cell_tokens: &[usize]) -> (usize, f32) {
        let mut weights: HashMap<usize, f32> = HashMap::new();
        let mut total_weight = 0.0f32;

        for (i, &t) in cell_tokens.iter().enumerate() {
            let conf = self.cell_confidence(i);
            *weights.entry(t).or_insert(0.0) += conf;
            total_weight += conf;
        }

        let (&winner, &weight) = weights.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((&0, &0.0));

        let consensus = if total_weight > 0.0 { weight / total_weight } else { 0.0 };
        (winner, consensus)
    }

    /// Leader-follow: cell 0 (lowest temperature) decides, others confirm
    fn vote_leader_follow(&self, cell_tokens: &[usize]) -> (usize, f32) {
        let leader_token = cell_tokens[0];
        let agree_count = cell_tokens.iter().filter(|&&t| t == leader_token).count();
        let consensus = agree_count as f32 / cell_tokens.len() as f32;
        (leader_token, consensus)
    }

    /// Estimate confidence for a cell based on its pending logits
    fn cell_confidence(&self, cell_idx: usize) -> f32 {
        let logits = &self.pending_logits[cell_idx];
        if logits.is_empty() { return 0.5; }

        // Max-softmax probability as confidence
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let max_prob = exps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if sum > 0.0 { max_prob / sum } else { 0.5 }
    }

    /// Generate tokens using ensemble voting
    pub fn generate(&mut self, prompt: &[usize], max_new: usize) -> (Vec<usize>, SpecStats) {
        let start = Instant::now();

        // Process prompt
        self.process_prompt(prompt);

        let mut output = prompt.to_vec();

        for _ in 0..max_new {
            let (token, _consensus) = self.generate_token();
            output.push(token);

            // EOS check
            if token == 0 { break; }
        }

        self.stats.total_time_ms = start.elapsed().as_millis() as u64;

        (output, self.stats.clone())
    }

    /// Generate with k-token speculative drafting
    /// Phase 2: draft k tokens, verify with ensemble, accept prefix on consensus
    pub fn generate_speculative(&mut self, prompt: &[usize], max_new: usize) -> (Vec<usize>, SpecStats) {
        let start = Instant::now();
        let k = self.config.draft_tokens;

        // Process prompt
        self.process_prompt(prompt);

        let mut output = prompt.to_vec();
        let mut generated = 0;

        while generated < max_new {
            // 1. Draft k tokens using cell 0 (leader, lowest temp)
            let draft_tokens = self.draft_k_tokens(k, &mut output);
            let draft_len = draft_tokens.len();

            // 2. Verify draft with all cells
            let (accepted, _rejected_at) = self.verify_draft(&draft_tokens, &output);

            // 3. Accept confirmed tokens
            for (i, &t) in draft_tokens.iter().enumerate() {
                if i < accepted {
                    output.push(t);
                    generated += 1;
                    self.stats.total_tokens += 1;
                    self.stats.accepted_draft += 1;
                    if t == 0 || generated >= max_new { break; }
                } else {
                    self.stats.rejected_draft += 1;
                }
            }

            // 4. If rejected, rollback and resample
            if accepted < draft_len {
                // Rollback states: need to replay the sequence
                // For simplicity, we resample from current position using voting
                let (voted, _) = self.generate_token();
                output.push(voted);
                generated += 1;
                if voted == 0 || generated >= max_new { break; }
            }

            self.stats.rounds += 1;
        }

        self.stats.total_time_ms = start.elapsed().as_millis() as u64;
        (output, self.stats.clone())
    }

    /// Draft k tokens using cell 0 (greedy/low temperature)
    fn draft_k_tokens(&mut self, k: usize, _output: &mut Vec<usize>) -> Vec<usize> {
        let mut draft = Vec::with_capacity(k);
        let state = &mut self.states[0];

        for _ in 0..k {
            let logits = &self.pending_logits[0];
            let token = sample_top_p(logits, 0.0, self.config.top_p); // greedy
            draft.push(token);

            if token == 0 { break; }

            // Advance cell 0's state
            self.pending_logits[0] = self.model.forward_token(token, state);
        }

        draft
    }

    /// Verify draft tokens with all cells
    /// Returns (accepted_count, first_rejection_index)
    fn verify_draft(&mut self, draft: &[usize], _output: &[usize]) -> (usize, usize) {
        let n = self.config.n_cells;

        for (pos, &draft_token) in draft.iter().enumerate() {
            // Get all cells' preference at this position
            let mut agreements = 0;

            for cell_idx in 1..n {
                // Cell's pending_logits are from the last processed token
                // Sample and check if it matches draft
                let cell_choice = self.sample_from_logits(&self.pending_logits[cell_idx], cell_idx);
                if cell_choice == draft_token {
                    agreements += 1;
                }
            }

            // Calculate consensus (excluding cell 0 which drafted)
            let consensus = (agreements + 1) as f32 / n as f32;

            if consensus < self.config.min_consensus {
                return (pos, pos);
            }

            // Advance all cells' states with the accepted draft token
            for cell_idx in 0..n {
                self.pending_logits[cell_idx] = self.model.forward_token(draft_token, &mut self.states[cell_idx]);
            }
        }

        (draft.len(), draft.len())
    }

    /// Compare: generate with single cell (baseline)
    pub fn generate_baseline(&self, prompt: &[usize], max_new: usize, temperature: f32) -> (Vec<usize>, u64) {
        let start = Instant::now();

        let mut state = RwkvModelState::new();
        let mut all_ids = prompt.to_vec();

        // Process prompt
        let mut last_logits = Vec::new();
        for &id in prompt {
            last_logits = self.model.forward_token(id, &mut state);
        }

        // Generate
        for _ in 0..max_new {
            let next = sample_top_p(&last_logits, temperature, self.config.top_p);
            all_ids.push(next);
            if next == 0 { break; }
            last_logits = self.model.forward_token(next, &mut state);
        }

        (all_ids, start.elapsed().as_millis() as u64)
    }

    /// Reset engine state
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
        for logits in &mut self.pending_logits {
            logits.clear();
        }
        self.stats = SpecStats { cell_hits: vec![0; self.config.n_cells], ..Default::default() };
    }
}

// ===================== Sampling =====================

/// Sample with temperature and top-p
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    if logits.is_empty() { return 0; }

    // Greedy
    if temperature < 1e-6 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();

    // Softmax
    let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_s: Vec<f32> = scaled.iter().map(|v| (v - max_s).exp()).collect();
    let sum_exp: f32 = exp_s.iter().sum();
    let probs: Vec<f32> = exp_s.iter().map(|v| v / sum_exp).collect();

    // Top-p filtering
    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut top: Vec<(usize, f32)> = Vec::new();
    for p in &pairs {
        if cumsum >= top_p { break; }
        cumsum += p.1;
        top.push(*p);
    }

    // Renormalize and sample
    let total: f32 = top.iter().map(|(_, p)| p).sum();
    if total <= 0.0 { return pairs.first().map(|(i, _)| *i).unwrap_or(0); }

    use rand::Rng;
    let r: f32 = rand::thread_rng().gen();
    let mut acc = 0.0f32;
    for (i, p) in top {
        acc += p / total;
        if r <= acc { return i; }
    }

    pairs.first().map(|(i, _)| *i).unwrap_or(0)
}

// ===================== Tests =====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.n_cells, 3);
        assert_eq!(cfg.temperatures.len(), 3);
        assert!(cfg.temperatures[0] < cfg.temperatures[1]);
    }

    #[test]
    fn test_config_with_cells() {
        let cfg = SpeculativeConfig::with_cells(5);
        assert_eq!(cfg.n_cells, 5);
        assert_eq!(cfg.temperatures.len(), 5);
    }

    #[test]
    fn test_vote_majority() {
        let cfg = SpeculativeConfig::default();
        let model = RwkvModel {
            emb: ndarray::Array2::zeros((VOCAB, 768)),
            layers: vec![],
            ln_out_w: vec![1.0; 768],
            ln_out_b: vec![0.0; 768],
            head: ndarray::Array2::zeros((VOCAB, 768)),
        };
        let engine = SpeculativeEngine::new(model, cfg);

        let (winner, consensus) = engine.vote_majority(&[1, 1, 2]);
        assert_eq!(winner, 1);
        assert!((consensus - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sample_top_p(&logits, 0.0, 0.9);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_sample_deterministic_with_low_temp() {
        let logits = vec![0.1, 10.0, 0.3];
        let token = sample_top_p(&logits, 0.001, 0.9);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_stats_consensus() {
        let stats = SpecStats {
            consensus_levels: vec![0.5, 0.75, 1.0],
            ..Default::default()
        };
        assert!((stats.avg_consensus() - 0.75).abs() < 0.01);
    }
}
