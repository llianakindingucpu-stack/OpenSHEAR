// SHEAR Aggregator — merges outputs from N parallel Cells into final token logits
// Strategy: learned routing weights (not fixed mean), competitive selection

use ndarray::Array1;
use serde::{Deserialize, Serialize};

// ===================== Aggregator Config =====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorConfig {
    pub vocab_size: usize,
    pub n_cells: usize,
    pub d_model: usize,
    /// How to combine Cell outputs
    pub strategy: CombineStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CombineStrategy {
    /// Weighted sum of logits (simplest, works well with diverse Cells)
    WeightedSum,
    /// Select the Cell with highest confidence (competitive)
    BestOfN,
    /// Rank-based: weight cells by confidence ranking
    RankBased,
}

impl AggregatorConfig {
    pub fn standard(vocab_size: usize, n_cells: usize) -> Self {
        Self {
            vocab_size,
            n_cells,
            d_model: 768,
            strategy: CombineStrategy::WeightedSum,
        }
    }

    pub fn tiny(vocab_size: usize, n_cells: usize) -> Self {
        Self {
            vocab_size,
            n_cells,
            d_model: 128,
            strategy: CombineStrategy::WeightedSum,
        }
    }
}

// ===================== Aggregator =====================

/// The Aggregator receives logits from N Cells and produces the final token distribution.
///
/// Key insight: we don't average outputs (that dumbs things down).
/// We weight them — Cells that are confident get more say.
/// Cells specialized for the input domain naturally dominate.
pub struct Aggregator {
    pub config: AggregatorConfig,
    /// Per-cell routing weights (learned, not fixed)
    /// Higher weight = more influence on final output
    pub cell_weights: Vec<f32>,
    /// Optional: per-cell specialization tags (e.g. "code", "math", "language")
    pub cell_tags: Vec<String>,
    /// Temperature for confidence estimation
    pub confidence_temp: f32,
}

impl Aggregator {
    pub fn new(config: AggregatorConfig) -> Self {
        let n = config.n_cells;
        // Start with equal weights
        let cell_weights = vec![1.0 / n as f32; n];
        let cell_tags = (0..n).map(|i| format!("cell-{}", i)).collect();

        Self {
            config,
            cell_weights,
            cell_tags,
            confidence_temp: 1.0,
        }
    }

    /// Create aggregator with uniform weights
    pub fn uniform(vocab_size: usize, n_cells: usize) -> Self {
        Self::new(AggregatorConfig::standard(vocab_size, n_cells))
    }

    /// Create aggregator with specific cell weights
    pub fn with_weights(vocab_size: usize, n_cells: usize, weights: Vec<f32>) -> Self {
        let mut agg = Self::new(AggregatorConfig::standard(vocab_size, n_cells));
        assert_eq!(weights.len(), n_cells, "weight count must match n_cells");
        let sum: f32 = weights.iter().sum();
        // Normalize weights
        agg.cell_weights = weights.iter().map(|w| w / sum).collect();
        agg
    }

    /// Set specialization tags for cells
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        assert_eq!(tags.len(), self.config.n_cells);
        self.cell_tags = tags;
        self
    }

    // ── Core: aggregate logits from N Cells ──────────────────

    /// Aggregate logits from multiple Cells.
    /// Each Cell produces [vocab_size] logits.
    pub fn aggregate(&self, cell_logits: &[Array1<f32>]) -> Array1<f32> {
        assert!(!cell_logits.is_empty(), "need at least 1 Cell output");
        assert_eq!(cell_logits.len(), self.config.n_cells,
            "expected {} Cell outputs, got {}", self.config.n_cells, cell_logits.len());

        match self.config.strategy {
            CombineStrategy::WeightedSum => self.weighted_sum(cell_logits),
            CombineStrategy::BestOfN => self.best_of_n(cell_logits),
            CombineStrategy::RankBased => self.rank_based(cell_logits),
        }
    }

    /// Weighted sum: final = Σ (w_i * logits_i)
    /// Simple and effective when Cells are diverse.
    fn weighted_sum(&self, cell_logits: &[Array1<f32>]) -> Array1<f32> {
        let vocab = self.config.vocab_size;
        let mut result = Array1::zeros(vocab);

        for (i, logits) in cell_logits.iter().enumerate() {
            let w = self.cell_weights[i];
            for j in 0..vocab {
                result[j] += w * logits[j];
            }
        }

        result
    }

    /// Best-of-N: pick the Cell with highest max-confidence.
    /// "Competitive selection" — the most confident Cell wins.
    fn best_of_n(&self, cell_logits: &[Array1<f32>]) -> Array1<f32> {
        let mut best_idx = 0;
        let mut best_conf = f32::NEG_INFINITY;

        for (i, logits) in cell_logits.iter().enumerate() {
            let conf = self.confidence(logits);
            if conf > best_conf {
                best_conf = conf;
                best_idx = i;
            }
        }

        cell_logits[best_idx].clone()
    }

    /// Rank-based: weight Cells by their confidence ranking.
    /// Top Cell gets weight 1.0, second gets 0.5, third gets 0.33, etc.
    fn rank_based(&self, cell_logits: &[Array1<f32>]) -> Array1<f32> {
        let vocab = self.config.vocab_size;

        // Calculate confidence for each Cell
        let mut ranked: Vec<(usize, f32)> = cell_logits.iter().enumerate()
            .map(|(i, logits)| (i, self.confidence(logits)))
            .collect();

        // Sort by confidence descending
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Assign rank-based weights: 1/rank
        let mut weights = vec![0.0f32; cell_logits.len()];
        for (rank, (idx, _)) in ranked.iter().enumerate() {
            weights[*idx] = 1.0 / (rank as f32 + 1.0);
        }

        // Normalize
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() { *w /= sum; }
        }

        // Weighted sum with rank-based weights
        let mut result = Array1::zeros(vocab);
        for (i, logits) in cell_logits.iter().enumerate() {
            for j in 0..vocab {
                result[j] += weights[i] * logits[j];
            }
        }

        result
    }

    // ── Confidence estimation ──────────────────────────────

    /// Estimate confidence of a Cell's output.
    /// Uses max-softmax probability (higher = more confident).
    fn confidence(&self, logits: &Array1<f32>) -> f32 {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|v| (*v - max) / self.confidence_temp).map(|v| v.exp()).collect();
        let sum: f32 = exps.iter().sum();
        let max_exp = exps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        max_exp / sum
    }

    /// Dynamic weight adjustment based on confidence.
    /// Call this after each token to adaptively route.
    pub fn adaptive_weights(&self, cell_logits: &[Array1<f32>]) -> Vec<f32> {
        let confidences: Vec<f32> = cell_logits.iter()
            .map(|logits| self.confidence(logits))
            .collect();

        let sum: f32 = confidences.iter().sum();
        if sum > 0.0 {
            confidences.iter().map(|c| c / sum).collect()
        } else {
            vec![1.0 / cell_logits.len() as f32; cell_logits.len()]
        }
    }

    /// Aggregate with adaptive (confidence-based) weights.
    /// Better than fixed weights for heterogeneous Cells.
    pub fn aggregate_adaptive(&self, cell_logits: &[Array1<f32>]) -> Array1<f32> {
        let vocab = self.config.vocab_size;
        let weights = self.adaptive_weights(cell_logits);

        let mut result = Array1::zeros(vocab);
        for (i, logits) in cell_logits.iter().enumerate() {
            for j in 0..vocab {
                result[j] += weights[i] * logits[j];
            }
        }

        result
    }

    // ── Token selection ──────────────────────────────────────

    /// Pick the next token from aggregated logits (greedy).
    pub fn select_token_greedy(logits: &Array1<f32>) -> usize {
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Pick the next token with temperature sampling.
    pub fn select_token_sample(logits: &Array1<f32>, temperature: f32) -> usize {
        use rand::Rng;
        let scaled = if temperature > 0.0 {
            logits.mapv(|v| v / temperature)
        } else {
            logits.clone()
        };

        // Softmax
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|v| (*v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Sample
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum { return i; }
        }
        probs.len() - 1
    }
}

// ===================== SHEAR Engine =====================

/// The full SHEAR inference engine: N Cells + 1 Aggregator.
/// This is the top-level entry point for parallel inference.
pub struct ShearEngine {
    pub aggregator: Aggregator,
    /// Cell IDs (for routing and logging)
    pub cell_ids: Vec<String>,
}

impl ShearEngine {
    pub fn new(aggregator: Aggregator, cell_ids: Vec<String>) -> Self {
        assert_eq!(cell_ids.len(), aggregator.config.n_cells);
        Self { aggregator, cell_ids }
    }

    /// Run one inference step: collect logits from all Cells, aggregate, select token.
    pub fn step(
        &self,
        cell_logits: &[Array1<f32>],
        temperature: f32,
    ) -> usize {
        let aggregated = self.aggregator.aggregate(cell_logits);

        if temperature <= 0.0 {
            Aggregator::select_token_greedy(&aggregated)
        } else {
            Aggregator::select_token_sample(&aggregated, temperature)
        }
    }

    /// Run one step with adaptive weighting.
    pub fn step_adaptive(
        &self,
        cell_logits: &[Array1<f32>],
        temperature: f32,
    ) -> usize {
        let aggregated = self.aggregator.aggregate_adaptive(cell_logits);

        if temperature <= 0.0 {
            Aggregator::select_token_greedy(&aggregated)
        } else {
            Aggregator::select_token_sample(&aggregated, temperature)
        }
    }
}

// ===================== Tests =====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_weighted_sum() {
        let agg = Aggregator::new(AggregatorConfig {
            vocab_size: 100,
            n_cells: 2,
            d_model: 64,
            strategy: CombineStrategy::WeightedSum,
        });

        let logits_a = Array1::from_vec(vec![1.0; 100]);
        let logits_b = Array1::from_vec(vec![3.0; 100]);

        let result = agg.aggregate(&[logits_a, logits_b]);
        // Equal weights: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        for &v in result.iter() {
            assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {}", v);
        }
    }

    #[test]
    fn test_aggregator_best_of_n() {
        let agg = Aggregator::new(AggregatorConfig {
            vocab_size: 10,
            n_cells: 3,
            d_model: 64,
            strategy: CombineStrategy::BestOfN,
        });

        // Cell B has highest confidence (peak at index 5)
        let mut logits_a = Array1::zeros(10);
        let mut logits_b = Array1::zeros(10);
        let mut logits_c = Array1::zeros(10);
        logits_a[3] = 1.0;
        logits_b[5] = 5.0;  // most confident
        logits_c[7] = 2.0;

        let result = agg.aggregate(&[logits_a, logits_b, logits_c]);
        // Should pick Cell B's output
        assert_eq!(result[5], 5.0);
    }

    #[test]
    fn test_aggregator_rank_based() {
        let agg = Aggregator::new(AggregatorConfig {
            vocab_size: 10,
            n_cells: 2,
            d_model: 64,
            strategy: CombineStrategy::RankBased,
        });

        let mut logits_a = Array1::zeros(10);
        let mut logits_b = Array1::zeros(10);
        logits_a[0] = 1.0;   // less confident
        logits_b[0] = 10.0;  // more confident

        let result = agg.aggregate(&[logits_a, logits_b]);
        // Cell B should dominate but not exclusively
        assert!(result[0] > 5.0, "rank-based should favor confident Cell B");
    }

    #[test]
    fn test_aggregator_adaptive() {
        let agg = Aggregator::new(AggregatorConfig {
            vocab_size: 10,
            n_cells: 2,
            d_model: 64,
            strategy: CombineStrategy::WeightedSum,
        });

        let mut logits_a = Array1::zeros(10);
        let mut logits_b = Array1::zeros(10);
        logits_a[0] = 1.0;
        logits_b[0] = 10.0;

        let result = agg.aggregate_adaptive(&[logits_a, logits_b]);
        // Adaptive should give more weight to Cell B
        assert!(result[0] > 5.0, "adaptive should favor confident Cell B");
    }

    #[test]
    fn test_select_token_greedy() {
        let mut logits = Array1::zeros(10);
        logits[7] = 5.0;
        assert_eq!(Aggregator::select_token_greedy(&logits), 7);
    }

    #[test]
    fn test_confidence() {
        let agg = Aggregator::new(AggregatorConfig {
            vocab_size: 10,
            n_cells: 1,
            d_model: 64,
            strategy: CombineStrategy::WeightedSum,
        });

        // Sharp distribution → high confidence
        let mut sharp = Array1::zeros(10);
        sharp[0] = 10.0;
        let c1 = agg.confidence(&sharp);

        // Flat distribution → low confidence
        let flat = Array1::from_vec(vec![1.0; 10]);
        let c2 = agg.confidence(&flat);

        assert!(c1 > c2, "sharp should be more confident than flat");
    }

    #[test]
    fn test_shear_engine_step() {
        let agg = Aggregator::new(AggregatorConfig {
            vocab_size: 10,
            n_cells: 2,
            d_model: 64,
            strategy: CombineStrategy::WeightedSum,
        });
        let engine = ShearEngine::new(agg, vec!["A".into(), "B".into()]);

        let mut logits_a = Array1::zeros(10);
        let mut logits_b = Array1::zeros(10);
        logits_a[3] = 2.0;
        logits_b[3] = 3.0;  // index 3 wins

        let token = engine.step(&[logits_a, logits_b], 0.0); // greedy
        assert_eq!(token, 3);
    }

    #[test]
    fn test_custom_weights() {
        let agg = Aggregator::with_weights(100, 2, vec![0.8, 0.2]);
        assert!((agg.cell_weights[0] - 0.8).abs() < 1e-5);
        assert!((agg.cell_weights[1] - 0.2).abs() < 1e-5);
    }
}
