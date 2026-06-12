//! Hidden-state aggregator — replaces token-voting with geometric operations in ℝ^H.
//!
//! Key insight: each node returns the final hidden state (before head projection),
//! not a token. Aggregation happens in continuous hidden space, then we decode
//! via nearest-neighbor in the vocab embedding space.
//!
//! Pipeline:
//!   1. Collect {node_id, hidden_state[H]} from each node
//!   2. Compute pairwise cosine similarity → affinity matrix
//!   3. Weighted-average aligned states → agg_state[H]
//!   4. Project agg_state → logits via head projection
//!   5. Sample next token
//!
//! This eliminates the token-space bottleneck: no need to decode/rencode tokens
//! between nodes. Coordination happens in hidden space (O(H) per node).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types

/// A node's contribution to aggregation: its final hidden state and metadata.
#[derive(Clone, Debug)]
pub struct NodeContribution {
    pub node_id: String,
    /// Hidden state after processing the full prompt — shape [H].
    pub hidden_state: Vec<f32>,
    /// Confidence score from local inference [0, 1].
    pub confidence: f32,
}

impl NodeContribution {
    pub fn new(node_id: String, hidden_state: Vec<f32>, confidence: f32) -> Self {
        Self { node_id, hidden_state, confidence }
    }
}

/// Aggregation config.
#[derive(Clone, Debug)]
pub struct HiddenAggConfig {
    /// Hidden dimension (must match model). 1024 for RWKV-4-World-430M.
    pub hidden_dim: usize,
    /// Cosine-similarity threshold below which a node is considered outlier.
    pub sim_threshold: f32,
    /// Temperature for sampling from aggregated logits.
    pub temperature: f32,
}

impl Default for HiddenAggConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1024,
            sim_threshold: 0.7,
            temperature: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Core algorithms

/// Cosine similarity between two vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    dot / (norm_a * norm_b)
}

/// Build pairwise cosine-similarity matrix among contributions.
pub fn build_affinity_matrix(contributions: &[NodeContribution]) -> Vec<Vec<f32>> {
    let n = contributions.len();
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = if i == j { 1.0 } else { cosine_sim(&contributions[i].hidden_state, &contributions[j].hidden_state) };
        }
    }
    mat
}

/// Weighted-average aggregation: nodes with higher similarity to the centroid
/// get higher weight. Returns the aggregated hidden state vector.
pub fn aggregate_weighted(contributions: &[NodeContribution], sim_threshold: f32) -> Vec<f32> {
    if contributions.is_empty() {
        return vec![];
    }
    if contributions.len() == 1 {
        return contributions[0].hidden_state.clone();
    }

    let h = contributions[0].hidden_state.len();

    // 1. Compute centroid (simple mean)
    let mut centroid = vec![0.0_f32; h];
    for c in contributions {
        for (i, v) in c.hidden_state.iter().enumerate() {
            centroid[i] += v;
        }
    }
    let n = contributions.len() as f32;
    for v in &mut centroid {
        *v /= n;
    }

    // 2. Compute each node's similarity to centroid
    let mut weights: Vec<f32> = contributions
        .iter()
        .map(|c| cosine_sim(&c.hidden_state, &centroid).max(0.0))
        .collect();

    // 3. Normalise — nodes below threshold get zero weight
    let max_w = weights.iter().cloned().fold(0.0_f32, f32::max);
    if max_w > 0.0 {
        for w in &mut weights {
            let raw = *w / max_w;
            *w = if raw >= sim_threshold { raw } else { 0.0 };
        }
    }
    let sum_w: f32 = weights.iter().sum();
    if sum_w < 1e-6 {
        // Fall back to equal weight
        for w in &mut weights { *w = 1.0 / n; }
    } else {
        for w in &mut weights { *w /= sum_w; }
    }

    // 4. Weighted sum
    let mut agg = vec![0.0_f32; h];
    for (c, &w) in contributions.iter().zip(weights.iter()) {
        for (i, v) in c.hidden_state.iter().enumerate() {
            agg[i] += w * v;
        }
    }
    agg
}

/// Decode from aggregated hidden state using the language-model head matrix.
/// logits = head @ agg_state  (shape [V, H] @ [H] → [V])
pub fn decode_hidden_state(
    agg_state: &[f32],
    head: &ndarray::Array2<f32>,
) -> Vec<f32> {
    let vocab = head.nrows();
    let mut logits = vec![0.0_f32; vocab];
    for (oi, row) in head.rows().into_iter().enumerate() {
        // Clamp to prevent f32 overflow
        let mut s = 0.0_f32;
        for (wi, xi) in row.iter().zip(agg_state.iter()) {
            let prod = wi * xi;
            s += if prod.is_nan() { 0.0 } else { prod.clamp(-1e20, 1e20) };
        }
        logits[oi] = s;
    }
    logits
}

// ---------------------------------------------------------------------------
// Main aggregator

/// Hidden-state aggregator.
pub struct HiddenStateAggregator {
    cfg: HiddenAggConfig,
}

impl HiddenStateAggregator {
    pub fn new(cfg: HiddenAggConfig) -> Self {
        Self { cfg }
    }

    /// Aggregate multiple node hidden states → logits over vocab.
    /// Returns (aggregated_logits, per_node_similarities).
    pub fn aggregate(
        &self,
        contributions: &[NodeContribution],
        head: &ndarray::Array2<f32>,
    ) -> (Vec<f32>, HashMap<String, f32>) {
        assert!(
            contributions.iter().all(|c| c.hidden_state.len() == self.cfg.hidden_dim),
            "hidden state dimension mismatch"
        );

        let agg_state = aggregate_weighted(contributions, self.cfg.sim_threshold);
        let logits = decode_hidden_state(&agg_state, head);
        let mat = build_affinity_matrix(contributions);

        // Per-node similarity to centroid (diagonal of affinity matrix gives self-similarity = 1)
        let mut sims = HashMap::new();
        for (i, c) in contributions.iter().enumerate() {
            // similarity to centroid ≈ average similarity to all others + self
            let avg: f32 = mat[i].iter().sum::<f32>() / mat[i].len() as f32;
            sims.insert(c.node_id.clone(), avg);
        }

        (logits, sims)
    }

    /// Top-k tokens from logits (greedy decode).
    pub fn topk_tokens(&self, logits: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut pairs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(k);
        pairs
    }
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(id: &str, seed: f32) -> NodeContribution {
        let h = 1024;
        NodeContribution::new(
            id.to_string(),
            (0..h).map(|i| (i as f32 * seed).sin()).collect(),
            0.9,
        )
    }

    #[test]
    fn test_cosine_sim_identical() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let s = cosine_sim(&a, &b);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let s = cosine_sim(&a, &b);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_affinity_matrix() {
        let c = vec![make_state("a", 1.0), make_state("b", 2.0)];
        let mat = build_affinity_matrix(&c);
        assert_eq!(mat.len(), 2);
        assert!((mat[0][0] - 1.0).abs() < 1e-6);
        assert!((mat[0][1] - mat[1][0]).abs() < 1e-6); // symmetric
    }

    #[test]
    fn test_aggregate_single_node() {
        let c = vec![make_state("a", 1.0)];
        let agg = aggregate_weighted(&c, 0.5);
        assert_eq!(agg.len(), 1024);
    }

    #[test]
    fn test_aggregate_two_similar_nodes() {
        let mut a = make_state("a", 1.0);
        let mut b = make_state("b", 1.0);
        // b = a + tiny noise
        for (i, v) in b.hidden_state.iter_mut().enumerate() {
            *v = a.hidden_state[i] + (i as f32 * 0.001);
        }
        b.confidence = 0.95;
        let agg = aggregate_weighted(&[a, b], 0.7);
        assert_eq!(agg.len(), 1024);
    }

    #[test]
    fn test_aggregator_new() {
        let cfg = HiddenAggConfig::default();
        assert_eq!(cfg.hidden_dim, 1024);
        assert_eq!(cfg.sim_threshold, 0.7);
    }
}
