//! SHEAR Speculative Decoding Engine — Phase 2: Domain-Aware Cell Routing
//!
//! Phase 1: Temperature diversity ensemble (all cells same domain, different temps)
//! Phase 2: Domain-aware routing (cells specialize by domain role)
//!
//! Core idea: Each cell has a Domain specialization. When a prompt is classified
//! as Code/Math/Reasoning/Dialogue/General, specialist cells get higher vote
//! weight. Draft cells draft k tokens; verifier cells confirm/reject.

use crate::rwkv_model::{RwkvModel, RwkvModelState};
use std::collections::HashMap;
use std::time::Instant;
use rayon::prelude::*;

// ============================================================================
// Domain Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    Code, Dialogue, Reasoning, Math, General,
}

impl Domain {
    pub fn label(self) -> &'static str {
        match self {
            Domain::Code      => "code",
            Domain::Dialogue  => "dialogue",
            Domain::Reasoning => "reasoning",
            Domain::Math      => "math",
            Domain::General   => "general",
        }
    }

    /// Lightweight keyword-based domain detector.
    /// In Phase 3 this will be replaced by a lightweight classifier head.
    pub fn detect(prompt: &str) -> Self {
        let p = prompt.to_lowercase();
        let code_kw   = ["def ", "fn ", "class ", "fn(", "import ", "const ", "struct ",
                         "pub fn", "for i in", "return ", "//", "/*", "print(", "print!("];
        let math_kw   = ["=", "solve for", "derivative", "integral", "matrix", "sqrt",
                         "sin", "cos", "theta", "log_", "proof", "theorem", "lim "];
        let reason_kw = ["think", "reason", "step", "why", "because", "therefore",
                         "thus", "since", "conclude", "hypothesis"];
        let dial_kw   = ["hello", "hi ", "how are", "what is", "what are", "?"];
        let code_c   = code_kw.iter().filter(|kw| p.contains(*kw)).count();
        let math_c   = math_kw.iter().filter(|kw| p.contains(*kw)).count();
        let reason_c = reason_kw.iter().filter(|kw| p.contains(*kw)).count();
        let dial_c   = dial_kw.iter().filter(|kw| p.contains(*kw)).count();
        let mx = code_c.max(math_c).max(reason_c).max(dial_c);
        if mx == 0 { return Domain::General; }
        if code_c   == mx { return Domain::Code; }
        if math_c   == mx { return Domain::Math; }
        if reason_c == mx { return Domain::Reasoning; }
        if dial_c   == mx { return Domain::Dialogue; }
        Domain::General
    }

    /// Priority bonus for specialist weighting (higher = more preferred).
    pub fn priority(self) -> u8 {
        match self {
            Domain::Code      => 4,
            Domain::Math      => 3,
            Domain::Reasoning => 2,
            Domain::Dialogue  => 1,
            Domain::General   => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CellRole {
    pub domain: Domain,
    pub weight: f32,
    pub is_drafter: bool,
}

impl CellRole {
    pub fn general() -> Self {
        Self { domain: Domain::General, weight: 1.0, is_drafter: false }
    }
    pub fn specialist(domain: Domain, is_drafter: bool) -> Self {
        Self { domain, weight: 1.0, is_drafter }
    }
}

// ============================================================================
// Config & VoteStrategy
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub n_cells: usize,
    pub temperatures: Vec<f32>,
    pub cell_roles: Vec<CellRole>,
    pub strategy: VoteStrategy,
    pub min_consensus: f32,
    pub top_p: f32,
    pub draft_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VoteStrategy {
    Majority,
    ConfidenceWeighted,
    LeaderFollow,
    /// Phase 2: weight votes by cell domain relevance to detected task
    DomainRouting,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            n_cells: 4,
            temperatures: vec![0.3, 0.6, 0.9, 1.2],
            cell_roles: vec![
                CellRole::specialist(Domain::Code, true),       // cell 0: code drafter
                CellRole::specialist(Domain::Reasoning, true),   // cell 1: reasoning drafter
                CellRole::specialist(Domain::Math, false),       // cell 2: math verifier
                CellRole::general(),                            // cell 3: general fallback
            ],
            strategy: VoteStrategy::DomainRouting,
            min_consensus: 0.5,
            top_p: 0.9,
            draft_tokens: 4,
        }
    }
}

impl SpeculativeConfig {
    pub fn with_cells(n: usize) -> Self {
        let temps: Vec<f32> = (0..n).map(|i| 0.3 + i as f32 * 0.3).collect();
        let roles: Vec<CellRole> = (0..n).map(|i| match i {
            0 => CellRole::specialist(Domain::Code, true),
            1 => CellRole::specialist(Domain::Reasoning, true),
            2 => CellRole::specialist(Domain::Math, false),
            _ => CellRole::general(),
        }).collect();
        Self {
            n_cells: n, temperatures: temps, cell_roles: roles,
            strategy: VoteStrategy::DomainRouting,
            min_consensus: 0.5, top_p: 0.9, draft_tokens: 4,
        }
    }

    pub fn domain_routing(n_cells: usize) -> Self {
        let n = n_cells.max(2);
        // Always: code + reasoning + general
        // Plus math if n > 2, plus dialogue if n > 3
        // Total: always 2 drafters + 1 general + optional math + optional dialogue
        // For n=4: code + reasoning + general + math = 4
        // For n=5: code + reasoning + general + math + dialogue = 5
        let mut roles = vec![
            CellRole::specialist(Domain::Code, true),       // drafter
            CellRole::specialist(Domain::Reasoning, true),  // drafter
            CellRole::general(),                            // general fallback
        ];
        if n > 2 { roles.insert(2, CellRole::specialist(Domain::Math, false)); }
        if n > 4 { roles.push(CellRole::specialist(Domain::Dialogue, false)); }
        Self {
            n_cells: roles.len(),
            temperatures: (0..roles.len()).map(|i| 0.3 + i as f32 * 0.3).collect(),
            cell_roles: roles,
            strategy: VoteStrategy::DomainRouting,
            min_consensus: 0.5, top_p: 0.9, draft_tokens: 4,
        }
    }
}

// ============================================================================
// Stats
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct SpecStats {
    pub total_tokens: usize,
    pub rounds: usize,
    pub consensus_levels: Vec<f32>,
    pub cell_hits: Vec<usize>,
    pub domain_hits: HashMap<Domain, usize>,
    pub total_time_ms: u64,
    pub accepted_draft: usize,
    pub rejected_draft: usize,
    pub detected_domain: Option<Domain>,
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

// ============================================================================
// Domain Voter (Phase 2 core)
// ============================================================================

struct DomainVoter;

impl DomainVoter {
    /// Compute domain-relevance weight for a cell given the detected domain.
    /// Specialist matching detected domain: full weight + priority bonus (up to 2.0x).
    /// Non-matching: generalist fallback (0.3x).
    fn domain_weight(role: &CellRole, detected: Domain) -> f32 {
        if role.domain == detected {
            (role.weight + role.domain.priority() as f32 * 0.05).min(2.0)
        } else {
            0.3 * role.weight
        }
    }

    /// Domain-aware vote: weight each cell's choice by (domain_relevance × confidence).
    fn vote_domain(
        cell_tokens: &[usize],
        confidences: &[f32],
        roles: &[CellRole],
        detected: Domain,
    ) -> (usize, f32) {
        let mut weights: HashMap<usize, f32> = HashMap::new();
        let mut total = 0.0f32;
        for (i, &tok) in cell_tokens.iter().enumerate() {
            let dw = Self::domain_weight(&roles[i], detected);
            let conf = confidences.get(i).copied().unwrap_or(0.5);
            let w = dw * conf;
            *weights.entry(tok).or_insert(0.0) += w;
            total += w;
        }
        let (&winner, &win_w) = weights.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((&0, &0.0));
        (winner, if total > 0.0 { win_w / total } else { 0.0 })
    }
}

// ============================================================================
// Domain-Speculative Engine (Phase 2)
// ============================================================================

/// Phase 2 Speculative Engine with domain-aware cell routing.
///
/// Key differences from Phase 1:
/// - `cell_roles`: each cell has a domain specialization
/// - `detected_domain`: prompt domain is classified once at setup
/// - `DomainRouting` strategy: votes weighted by domain relevance
/// - Speculative execution: drafter cells draft, verifier cells confirm
pub struct DomainSpeculativeEngine {
    pub model: RwkvModel,
    pub config: SpeculativeConfig,
    /// Per-cell model states (independent RNN hidden states)
    states: Vec<RwkvModelState>,
    /// Last logits per cell (output of most recent forward pass)
    pending_logits: Vec<Vec<f32>>,
    /// Detected domain for current generation session
    detected_domain: Domain,
    /// Domain-specific weights computed from detected_domain
    domain_weights: Vec<f32>,
    stats: SpecStats,
}

impl DomainSpeculativeEngine {
    pub fn new(model: RwkvModel, config: SpeculativeConfig) -> Self {
        let n = config.n_cells;
        let states = (0..n).map(|_| RwkvModelState::new()).collect();
        let pending_logits = vec![vec![]; n];
        let cell_hits = vec![0; n];
        let detected_domain = Domain::General;
        let domain_weights = (0..n)
            .map(|i| DomainVoter::domain_weight(&config.cell_roles[i], detected_domain))
            .collect();
        Self {
            model, config, states, pending_logits,
            detected_domain, domain_weights,
            stats: SpecStats {
                cell_hits,
                detected_domain: Some(detected_domain),
                ..Default::default()
            },
        }
    }

    /// Detect domain from prompt text and update domain weights.
    /// Called once at the start of each generation session.
    pub fn detect_and_prepare(&mut self, prompt_text: &str) {
        self.detected_domain = Domain::detect(prompt_text);
        self.stats.detected_domain = Some(self.detected_domain);
        tracing::info!(
            "[Spec] domain={} prompt=\"{:.30}...\"",
            self.detected_domain.label(),
            prompt_text.chars().take(30).collect::<String>()
        );
        self.domain_weights = (0..self.config.n_cells)
            .map(|i| DomainVoter::domain_weight(&self.config.cell_roles[i], self.detected_domain))
            .collect();
    }

    pub fn vocab_size(&self) -> usize {
        self.model.emb.nrows()
    }

    /// Process prompt tokens through all cells in parallel (rayon).
    pub fn process_prompt(&mut self, tokens: &[usize]) {
        let states = std::mem::take(&mut self.states);
        let model = &self.model;
        let (new_states, new_logits): (Vec<_>, Vec<_>) = states
            .into_par_iter()
            .map(|mut state| {
                state.reset();
                let mut last = Vec::new();
                for &id in tokens {
                    last = model.forward_token(id, &mut state);
                }
                (state, last)
            })
            .unzip();
        self.states = new_states;
        self.pending_logits = new_logits;
    }

    fn sample_from_logits(&self, logits: &[f32], cell_idx: usize) -> usize {
        let temp = self.config.temperatures.get(cell_idx).copied().unwrap_or(0.8);
        sample_top_p(logits, temp, self.config.top_p)
    }

    /// Estimate softmax confidence for a cell's pending logits (max-prob approximation).
    fn cell_confidence(&self, cell_idx: usize) -> f32 {
        let logits = &self.pending_logits[cell_idx];
        if logits.is_empty() { return 0.5; }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 { return 0.5; }
        exps.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / sum
    }

    /// Generate one token via domain-aware ensemble voting.
    pub fn generate_token(&mut self) -> (usize, f32) {
        let n = self.config.n_cells;

        // 1. Each cell samples from its pending logits
        let cell_tokens: Vec<usize> = (0..n)
            .map(|i| self.sample_from_logits(&self.pending_logits[i], i))
            .collect();

        // 2. Compute per-cell confidences
        let confidences: Vec<f32> = (0..n).map(|i| self.cell_confidence(i)).collect();

        // 3. Vote using domain-aware routing
        let (voted, consensus) = match self.config.strategy {
            VoteStrategy::Majority          => self.vote_majority(&cell_tokens),
            VoteStrategy::ConfidenceWeighted => self.vote_confidence(&cell_tokens, &confidences),
            VoteStrategy::LeaderFollow      => self.vote_leader_follow(&cell_tokens),
            VoteStrategy::DomainRouting      => DomainVoter::vote_domain(
                &cell_tokens, &confidences, &self.config.cell_roles, self.detected_domain),
        };

        // 4. Advance all cells with voted token (parallel)
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

        // 5. Track cell hits by domain
        for (i, &t) in cell_tokens.iter().enumerate() {
            if t == voted {
                self.stats.cell_hits[i] += 1;
                *self.stats.domain_hits
                    .entry(self.config.cell_roles[i].domain)
                    .or_insert(0) += 1;
            }
        }

        self.stats.total_tokens += 1;
        self.stats.rounds += 1;
        self.stats.consensus_levels.push(consensus);

        (voted, consensus)
    }

    fn vote_majority(&self, cell_tokens: &[usize]) -> (usize, f32) {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &t in cell_tokens { *counts.entry(t).or_insert(0) += 1; }
        // Stable tie-break: highest count wins; smallest token id on ties.
        // Using (count, -token_id) so max_by_key gives highest count first,
        // then lowest token id as tiebreaker.
        let (&winner, &count) = counts.iter()
            .max_by_key(|(tok, cnt)| (*cnt, usize::MAX - *tok))
            .unwrap_or((&0, &0));
        (winner, count as f32 / cell_tokens.len() as f32)
    }

    fn vote_confidence(&self, cell_tokens: &[usize], confidences: &[f32]) -> (usize, f32) {
        let mut weights: HashMap<usize, f32> = HashMap::new();
        let mut total = 0.0f32;
        for (i, &t) in cell_tokens.iter().enumerate() {
            let w = confidences.get(i).copied().unwrap_or(0.5);
            *weights.entry(t).or_insert(0.0) += w;
            total += w;
        }
        let (&winner, &win_w) = weights.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((&0, &0.0));
        (winner, if total > 0.0 { win_w / total } else { 0.0 })
    }

    fn vote_leader_follow(&self, cell_tokens: &[usize]) -> (usize, f32) {
        let leader = cell_tokens[0];
        let agree = cell_tokens.iter().filter(|&&t| t == leader).count();
        (leader, agree as f32 / cell_tokens.len() as f32)
    }

    // =========================================================================
    // Phase 2: Speculative Execution
    // =========================================================================

    /// Generate with domain-aware speculative decoding.
    /// Drafter cells produce k tokens; verifier cells confirm/reject each.
    pub fn generate_speculative(
        &mut self,
        prompt_text: &str,
        prompt_tokens: &[usize],
        max_new: usize,
    ) -> (Vec<usize>, SpecStats) {
        let start = Instant::now();

        self.detect_and_prepare(prompt_text);
        self.process_prompt(prompt_tokens);

        let mut output = prompt_tokens.to_vec();
        let mut generated = 0;

        while generated < max_new {
            let k = self.config.draft_tokens.min(max_new - generated);
            let draft = self.draft_k_tokens(k);
            let (accepted, _) = self.verify_speculative(&draft);

            // Accept confirmed prefix
            for (pos, &t) in draft.iter().enumerate() {
                if pos < accepted {
                    output.push(t);
                    generated += 1;
                    self.stats.total_tokens += 1;
                    self.stats.accepted_draft += 1;
                    if t == 0 || generated >= max_new { break; }
                } else {
                    self.stats.rejected_draft += 1;
                }
            }

            // Fall back to domain-voted token on rejection
            if accepted < draft.len() {
                let (voted, _) = self.generate_token();
                output.push(voted);
                generated += 1;
                self.stats.total_tokens += 1;
                self.stats.rejected_draft += 1;
                if voted == 0 || generated >= max_new { break; }
            }

            self.stats.rounds += 1;
        }

        self.stats.total_time_ms = start.elapsed().as_millis() as u64;
        (output, self.stats.clone())
    }

    /// Draft k tokens using the primary drafter cell (cell 0, code specialist, greedy).
    /// All cells' states advance in parallel during drafting.
    fn draft_k_tokens(&mut self, k: usize) -> Vec<usize> {
        let mut draft = Vec::with_capacity(k);
        for _ in 0..k {
            let logits = &self.pending_logits[0];
            let token = sample_top_p(logits, 0.0, self.config.top_p); // greedy
            draft.push(token);
            if token == 0 { break; }

            let states = std::mem::take(&mut self.states);
            let model = &self.model;
            let t = token;
            let (new_states, new_logits): (Vec<_>, Vec<_>) = states
                .into_par_iter()
                .map(|mut state| {
                    let logits = model.forward_token(t, &mut state);
                    (state, logits)
                })
                .unzip();
            self.states = new_states;
            self.pending_logits = new_logits;
        }
        draft
    }

    /// Verify draft tokens: each verifier cell checks each draft position.
    /// Returns (accepted_count, first_rejection_position).
    fn verify_speculative(&mut self, draft: &[usize]) -> (usize, usize) {
        let n = self.config.n_cells;
        if draft.is_empty() { return (0, 0); }

        for (pos, &draft_tok) in draft.iter().enumerate() {
            // Weighted vote across all cells
            let mut weights: HashMap<usize, f32> = HashMap::new();
            let mut total = 0.0f32;
            for cell_idx in 0..n {
                let logits = &self.pending_logits[cell_idx];
                let cell_tok = self.sample_from_logits(logits, cell_idx);
                let dw = *self.domain_weights.get(cell_idx).unwrap_or(&0.3);
                let conf = self.cell_confidence(cell_idx);
                *weights.entry(cell_tok).or_insert(0.0) += dw * conf;
                total += dw * conf;
            }

            let draft_w = weights.get(&draft_tok).copied().unwrap_or(0.0);
            let consensus = if total > 0.0 { draft_w / total } else { 0.0 };

            if consensus < self.config.min_consensus {
                return (pos, pos);
            }

            // Advance all cells with accepted token (already done in draft_k_tokens)
        }
        (draft.len(), draft.len())
    }

    /// Generate tokens using domain-aware voting (non-speculative).
    pub fn generate(
        &mut self,
        prompt_text: &str,
        prompt_tokens: &[usize],
        max_new: usize,
    ) -> (Vec<usize>, SpecStats) {
        let start = Instant::now();
        self.detect_and_prepare(prompt_text);
        self.process_prompt(prompt_tokens);

        let mut output = prompt_tokens.to_vec();
        for _ in 0..max_new {
            let (token, _) = self.generate_token();
            output.push(token);
            if token == 0 { break; }
        }

        self.stats.total_time_ms = start.elapsed().as_millis() as u64;
        (output, self.stats.clone())
    }

    /// Backward-compatible generate (Phase 1 signature: tokens-only, no domain detection).
    /// Detects domain as "general" for full ensemble.
    pub fn generate_phase1(&mut self, prompt_tokens: &[usize], max_new: usize) -> (Vec<usize>, SpecStats) {
        let start = Instant::now();
        self.process_prompt(prompt_tokens);
        let mut output = prompt_tokens.to_vec();
        for _ in 0..max_new {
            let (token, _) = self.generate_token();
            output.push(token);
            if token == 0 { break; }
        }
        self.stats.total_time_ms = start.elapsed().as_millis() as u64;
        (output, self.stats.clone())
    }

    /// Single-cell baseline generation (no ensemble, for comparison only).
    /// Runs one isolated cell through the prompt and generates with a fixed temperature.
    pub fn generate_baseline(&self, prompt_tokens: &[usize], max_new: usize, temperature: f32) -> (Vec<usize>, u64) {
        let start = Instant::now();
        let mut state = RwkvModelState::new();
        let mut all = prompt_tokens.to_vec();
        // Process prompt
        for &id in prompt_tokens {
            self.model.forward_token(id, &mut state);
        }
        // Generate
        for _ in 0..max_new {
            let logits = &self.pending_logits.get(0).cloned().unwrap_or_default();
            let next = sample_top_p(logits, temperature, self.config.top_p);
            all.push(next);
            if next == 0 { break; }
            self.model.forward_token(next, &mut state);
        }
        (all, start.elapsed().as_millis() as u64)
    }

    /// Reset engine state between generation sessions.
    pub fn reset(&mut self) {
        for s in &mut self.states { s.reset(); }
        for l in &mut self.pending_logits { l.clear(); }
        self.stats = SpecStats {
            cell_hits: vec![0; self.config.n_cells],
            detected_domain: Some(self.detected_domain),
            ..Default::default()
        };
    }
}

// ============================================================================
// Sampling Utilities
// ============================================================================

/// Sample from logits with temperature scaling + top-p truncation.
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    if logits.is_empty() { return 0; }

    // Greedy path
    if temperature < 1e-6 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();

    // Softmax
    let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_s: Vec<f32> = scaled.iter().map(|v| (v - max_s).exp()).collect();
    let sum_exp: f32 = exp_s.iter().sum();
    if sum_exp <= 0.0 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
    }
    let probs: Vec<f32> = exp_s.iter().map(|v| v / sum_exp).collect();

    // Top-p truncation
    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0f32;
    let top: Vec<(usize, f32)> = pairs
        .iter()
        .take_while(|(_, p)| {
            if cumsum >= top_p { false } else { cumsum += *p; true }
        })
        .cloned()
        .collect();

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 100;

    fn dummy_model() -> RwkvModel {
        RwkvModel {
            emb: ndarray::Array2::zeros((VOCAB, 768)),
            layers: vec![],
            ln_out_w: vec![1.0; 768],
            ln_out_b: vec![0.0; 768],
            head: ndarray::Array2::zeros((VOCAB, 768)),
        }
    }

    // ---- Domain detection ----

    #[test]
    fn test_domain_detect_code() {
        assert_eq!(Domain::detect("def foo():\n    return 42"), Domain::Code);
        assert_eq!(Domain::detect("fn main() {\n    println!(\"{}\");\n}"), Domain::Code);
        assert_eq!(Domain::detect("class MyClass:"), Domain::Code);
        assert_eq!(Domain::detect("import numpy as np\n"), Domain::Code);
    }

    #[test]
    fn test_domain_detect_math() {
        assert_eq!(Domain::detect("solve for x: 2x + 5 = 15"), Domain::Math);
        assert_eq!(Domain::detect("compute the integral of sin(x)"), Domain::Math);
        assert_eq!(Domain::detect("derivative of x^2 + 3x"), Domain::Math);
        assert_eq!(Domain::detect("prove that sqrt(2) is irrational"), Domain::Math);
    }

    #[test]
    fn test_domain_detect_reasoning() {
        assert_eq!(Domain::detect("think step by step about this"), Domain::Reasoning);
        assert_eq!(Domain::detect("because therefore thus conclude"), Domain::Reasoning);
        assert_eq!(Domain::detect("explain why this is the case"), Domain::Reasoning);
    }

    #[test]
    fn test_domain_detect_dialogue() {
        assert_eq!(Domain::detect("hello how are you today?"), Domain::Dialogue);
        assert_eq!(Domain::detect("what is the capital of France?"), Domain::Dialogue);
    }

    #[test]
    fn test_domain_detect_general() {
        assert_eq!(Domain::detect("tell me a story about dragons"), Domain::General);
        assert_eq!(Domain::detect("summarize the main points of this"), Domain::General);
    }

    #[test]
    fn test_domain_priority() {
        assert!(Domain::Code.priority() > Domain::General.priority());
        assert!(Domain::Math.priority() > Domain::Dialogue.priority());
        assert_eq!(Domain::General.priority(), 0);
    }

    // ---- Config ----

    #[test]
    fn test_config_default_has_domain_routing() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.strategy, VoteStrategy::DomainRouting);
        assert_eq!(cfg.cell_roles[0].domain, Domain::Code);
        assert!(cfg.cell_roles[0].is_drafter);
        assert_eq!(cfg.n_cells, 4);
    }

    #[test]
    fn test_config_domain_routing_factory() {
        let cfg = SpeculativeConfig::domain_routing(4);
        // domain_routing(4) = code(0) + reasoning(1) + math(2) + general(3) = 4 specialist slots
        // The factory adds: 2 hardcoded drafters + math if n>2 + dialogue if n>3 + general
        // With n=4: code + reasoning + math + general = 4 cells (dialogue skipped: 4 > 3)
        assert_eq!(cfg.n_cells, 4);
        assert_eq!(cfg.strategy, VoteStrategy::DomainRouting);
        assert!(cfg.cell_roles[0].is_drafter);  // code drafter
        assert!(cfg.cell_roles[1].is_drafter);  // reasoning drafter
    }

    #[test]
    fn test_config_domain_routing_5_cells() {
        // domain_routing(5): code + reasoning + math + general + dialogue = 5
        let cfg = SpeculativeConfig::domain_routing(5);
        assert_eq!(cfg.n_cells, 5);
        assert!(cfg.cell_roles[0].is_drafter);
        assert!(cfg.cell_roles[1].is_drafter);
        // dialogue added at position 4 (last) because n > 4
        assert_eq!(cfg.cell_roles[4].domain, Domain::Dialogue);
    }

    // ---- Domain voting ----

    #[test]
    fn test_domain_vote_weights_specialist() {
        let code_spec = CellRole::specialist(Domain::Code, true);
        let gen = CellRole::general();

        // Code specialist gets high weight for code domain
        let w_code = DomainVoter::domain_weight(&code_spec, Domain::Code);
        let w_gen  = DomainVoter::domain_weight(&gen, Domain::Code);
        assert!(w_code > w_gen);

        // Code specialist demoted for reasoning domain
        let w_code_r = DomainVoter::domain_weight(&code_spec, Domain::Reasoning);
        assert!(w_code_r < w_code);
    }

    #[test]
    fn test_vote_majority() {
        let engine = DomainSpeculativeEngine::new(dummy_model(), SpeculativeConfig::default());
        let (winner, c) = engine.vote_majority(&[1, 1, 2, 2]);
        assert_eq!(winner, 1);
        assert!((c - 0.5).abs() < 0.01);

        let (w2, c2) = engine.vote_majority(&[1, 1, 1, 2]);
        assert_eq!(w2, 1);
        assert!((c2 - 0.75).abs() < 0.01);
    }

    // ---- Sampling ----

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sample_top_p(&logits, 0.0, 0.9), 3);
    }

    #[test]
    fn test_sample_low_temp() {
        let logits = vec![0.1, 10.0, 0.3];
        assert_eq!(sample_top_p(&logits, 0.001, 0.9), 1);
    }

    // ---- Stats ----

    #[test]
    fn test_stats_consensus() {
        let stats = SpecStats {
            consensus_levels: vec![0.5, 0.75, 1.0],
            ..Default::default()
        };
        assert!((stats.avg_consensus() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_cell_roles() {
        let c = CellRole::specialist(Domain::Code, true);
        assert_eq!(c.domain, Domain::Code);
        assert!(c.is_drafter);
        assert_eq!(c.weight, 1.0);

        let g = CellRole::general();
        assert_eq!(g.domain, Domain::General);
        assert!(!g.is_drafter);
    }
}

// ============================================================================
// Backward-compatibility aliases (for existing binary targets)
// ============================================================================

/// Backward-compatible alias: Phase 1 ensemble engine → Phase 2 domain engine.
pub type SpeculativeEngine = DomainSpeculativeEngine;

/// Backward-compatible type alias for existing code that uses EnsembleSpeculativeConfig.
pub type EnsembleSpeculativeConfig = SpeculativeConfig;
