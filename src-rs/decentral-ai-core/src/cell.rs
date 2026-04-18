// SHEAR Cell — the core inference unit
// ~200M params: Embedding → TimeMix (linear recurrence) → SwiGLU FFN → Output
// No cross-Cell communication. No KV Cache. Fully stateless to other Cells.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

// ===================== Config =====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellConfig {
    pub vocab_size: usize,
    pub d_model: usize,       // hidden dimension
    pub d_ffn: usize,         // FFN intermediate size
    pub n_layers: usize,      // number of time-mix + ffn blocks
    pub head_size: usize,     // time-mix head dimension (RWKV divides d_model into heads)
    pub n_heads: usize,       // number of time-mix heads
    pub max_seq_len: usize,   // context window
}

impl CellConfig {
    /// Standard SHEAR Cell (~200M params)
    /// 50277 vocab × 768 d_model × 6 layers ≈ 200M
    pub fn standard() -> Self {
        Self {
            vocab_size: 50277,
            d_model: 768,
            d_ffn: 3072,
            n_layers: 6,
            head_size: 64,
            n_heads: 12,
            max_seq_len: 2048,
        }
    }

    /// Tiny cell for testing (~2M params)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1024,
            d_model: 128,
            d_ffn: 512,
            n_layers: 2,
            head_size: 32,
            n_heads: 4,
            max_seq_len: 128,
        }
    }

    pub fn total_params(&self) -> usize {
        // Embedding: vocab × d_model
        let emb = self.vocab_size * self.d_model;
        // Per layer: time-mix weights + ffn weights + layernorm
        // time-mix: Wk,Wv,Wo,Wq,Wr (each d_model × d_model) + time-mix decay/key/value/receptance (each d_model)
        // Simplified: ~8 * d_model^2 per layer for time-mix + 3 * d_model * d_ffn for ffn + layernorms
        let tm = 6 * self.d_model * self.d_model + 6 * self.d_model; // time-mix
        let ffn = 3 * self.d_model * self.d_ffn + 2 * self.d_ffn;     // SwiGLU + bias
        let ln = 4 * 2 * self.d_model;                                 // 4 layernorms per layer
        let per_layer = tm + ffn + ln;
        // Output head: d_model × vocab_size
        let out = self.d_model * self.vocab_size;
        emb + self.n_layers * per_layer + out
    }
}

// ===================== Weights =====================

/// All weights for a single Cell, stored as flat f32 vectors with shape info.
/// Loaded from safetensors at startup.
#[derive(Debug, Clone)]
pub struct CellWeights {
    pub config: CellConfig,
    /// name → (flat f32 data, shape)
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl CellWeights {
    pub fn new(config: CellConfig) -> Self {
        Self { config, tensors: HashMap::new() }
    }

    /// Get a tensor as Array2 (2D matrix). Panics if shape mismatches.
    pub fn get_2d(&self, name: &str) -> Array2<f32> {
        let (data, shape) = self.tensors.get(name)
            .unwrap_or_else(|| panic!("tensor '{}' not found", name));
        assert_eq!(shape.len(), 2, "tensor '{}' is not 2D: {:?}", name, shape);
        let rows = shape[0];
        let cols = shape[1];
        Array2::from_shape_vec((rows, cols), data.clone())
            .unwrap_or_else(|e| panic!("reshape '{}' ({:?}): {}", name, shape, e))
    }

    /// Get a tensor as Array1 (1D vector).
    pub fn get_1d(&self, name: &str) -> Array1<f32> {
        let (data, shape) = self.tensors.get(name)
            .unwrap_or_else(|| panic!("tensor '{}' not found", name));
        assert_eq!(shape.len(), 1, "tensor '{}' is not 1D: {:?}", name, shape);
        Array1::from_vec(data.clone())
    }

    /// Insert a tensor.
    pub fn set(&mut self, name: String, data: Vec<f32>, shape: Vec<usize>) {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "tensor '{}' data length {} != shape product {}", name, data.len(), expected);
        self.tensors.insert(name, (data, shape));
    }

    /// Parameter count
    pub fn param_count(&self) -> usize {
        self.tensors.values().map(|(d, _)| d.len()).sum()
    }
}

// ===================== Math Primitives =====================

/// Layer normalization (simplified, no learned bias by default)
pub fn layer_norm(x: &Array1<f32>, weight: &Array1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let mean = x.sum() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(v, w)| (v - mean) * inv_std * w).collect()
}

/// SiLU activation: x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    let sx = 1.0 / (1.0 + (-x).exp()); // sigmoid
    x * sx
}

/// Element-wise SiLU on Array1
pub fn silu_vec(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(silu)
}

/// Softmax (numerically stable)
pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    Array1::from_vec(exp.iter().map(|e| e / sum).collect())
}

// ===================== Time-Mix (RWKV-style Linear Recurrence) =====================

/// Per-head state for the time-mix layer.
/// Unlike KV Cache, this is a fixed-size vector that doesn't grow with sequence length.
#[derive(Debug, Clone)]
pub struct TimeMixState {
    /// Per-head: (aa, bb, pp) — running accumulators
    pub aa: Vec<f32>,  // [n_heads]
    pub bb: Vec<f32>,  // [n_heads]
    pub pp: Vec<f32>,  // [n_heads]
}

impl TimeMixState {
    pub fn new(n_heads: usize) -> Self {
        Self {
            aa: vec![0.0; n_heads],
            bb: vec![0.0; n_heads],
            pp: vec![0.0; n_heads],
        }
    }
}

/// Time-mix layer: linear recurrence with learned decay.
///
/// Formula (per head h):
///   w  = exp(-exp(decay_h) * (step + 1))
///   k  = (Wk @ x) + key_decay * prev_k
///   v  = (Wv @ x) + value_decay * prev_v
///   r  = sigmoid((Wr @ x) + receptance_decay * prev_r)
///   aa = w * prev_aa + k * v           (running numerator)
///   bb = w * prev_bb + v               (running denominator)
///   ww = w * prev_pp + k               (for output computation)
///   pp = prev_ww
///   output = r * (aa / bb)             (gated output)
///
/// Key insight: O(1) per token, no KV Cache growth.
pub struct TimeMixLayer {
    pub wk: Array2<f32>,  // [d_model, d_model] key projection
    pub wv: Array2<f32>,  // [d_model, d_model] value projection
    pub wr: Array2<f32>,  // [d_model, d_model] receptance projection
    pub wo: Array2<f32>,  // [d_model, d_model] output projection
    pub decay: Array1<f32>,    // [d_model] time decay
    pub key_decay: Array1<f32>,   // [d_model] key mixing
    pub value_decay: Array1<f32>, // [d_model] value mixing
    pub recept_decay: Array1<f32>,// [d_model] receptance mixing
    pub ln: (Array1<f32>, Array1<f32>), // layernorm weight + bias
}

impl TimeMixLayer {
    pub fn new(w: &CellWeights, layer_idx: usize) -> Self {
        let p = format!("tm.{}.", layer_idx);
        Self {
            wk: w.get_2d(&format!("{}wk.weight", p)),
            wv: w.get_2d(&format!("{}wv.weight", p)),
            wr: w.get_2d(&format!("{}wr.weight", p)),
            wo: w.get_2d(&format!("{}wo.weight", p)),
            decay: w.get_1d(&format!("{}decay", p)),
            key_decay: w.get_1d(&format!("{}key_decay", p)),
            value_decay: w.get_1d(&format!("{}value_decay", p)),
            recept_decay: w.get_1d(&format!("{}recept_decay", p)),
            ln: (
                w.get_1d(&format!("{}ln.weight", p)),
                w.get_1d(&format!("{}ln.bias", p)),
            ),
        }
    }

    /// Forward pass for one token.
    /// x: [d_model] input, state: mutable per-head state, step: position in sequence.
    pub fn forward(&self, x: &Array1<f32>, state: &mut TimeMixState, step: usize) -> Array1<f32> {
        let d = x.len();
        let head_size = d / state.aa.len();

        // Layer norm input
        let x_norm = layer_norm(x, &self.ln.0, 1e-5);

        // Projections: [d_model] = [d_model, d_model]^T @ [d_model]
        let k = self.wk.t().dot(&x_norm);
        let v = self.wv.t().dot(&x_norm);
        let r = self.wr.t().dot(&x_norm);

        // Convert to owned vecs once (outside loop)
        let k_v = k.to_vec();
        let v_v = v.to_vec();
        let r_v = r.to_vec();

        // Process each head independently
        let mut output = vec![0.0f32; d];

        for h in 0..state.aa.len() {
            let start = h * head_size;
            let end = start + head_size;

            // Compute time decay
            let decay_val = (-self.decay[start].exp() * (step as f32 + 1.0)).exp();

            // Slice head portion from owned vecs
            let k_h = &k_v[start..end];
            let v_h = &v_v[start..end];
            let r_h = &r_v[start..end];

            // Simple head-level recurrence: accumulate (k * v) over time
            let kv_dot: f32 = k_h.iter().zip(v_h.iter()).map(|(a, b)| a * b).sum();
            let v_sum: f32 = v_h.iter().sum();

            let new_aa = decay_val * state.aa[h] + kv_dot;
            let new_bb = decay_val * state.bb[h] + v_sum;
            let new_pp = decay_val * state.pp[h] + k_h.iter().sum::<f32>();

            // Output: receptance * (accumulated_value / accumulated_weight)
            let denom = new_bb.abs().max(1e-8);
            let ratio = new_aa / denom;

            for j in start..end {
                output[j] = r_h[j - start] * ratio;
            }

            // Update state
            state.aa[h] = new_aa;
            state.bb[h] = new_bb;
            state.pp[h] = new_pp;
        }

        // Output projection
        let out = self.wo.t().dot(&Array1::from(output));

        // Residual connection
        &out + x
    }
}

// ===================== SwiGLU FFN =====================

pub struct SwiGLUFFN {
    pub w1: Array2<f32>,  // [d_model, d_ffn] gate projection
    pub w2: Array2<f32>,  // [d_ffn, d_model] down projection
    pub w3: Array2<f32>,  // [d_model, d_ffn] up projection
    pub ln: (Array1<f32>, Array1<f32>),
}

impl SwiGLUFFN {
    pub fn new(w: &CellWeights, layer_idx: usize) -> Self {
        let p = format!("ffn.{}.", layer_idx);
        Self {
            w1: w.get_2d(&format!("{}w1.weight", p)),
            w2: w.get_2d(&format!("{}w2.weight", p)),
            w3: w.get_2d(&format!("{}w3.weight", p)),
            ln: (
                w.get_1d(&format!("{}ln.weight", p)),
                w.get_1d(&format!("{}ln.bias", p)),
            ),
        }
    }

    /// SwiGLU(x) = (x @ W1 * silu(x @ W3)) @ W2
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let x_norm = layer_norm(x, &self.ln.0, 1e-5);

        // Up and gate projections
        let gate = self.w1.t().dot(&x_norm); // [d_ffn]
        let up = self.w3.t().dot(&x_norm);    // [d_ffn]

        // SwiGLU: gate * silu(up)
        let mut hidden = Vec::with_capacity(gate.len());
        for i in 0..gate.len() {
            hidden.push(gate[i] * silu(up[i]));
        }

        // Down projection
        let out = self.w2.t().dot(&Array1::from(hidden));

        // Residual connection
        &out + x
    }
}

// ===================== Cell =====================

/// A single SHEAR Cell — the fundamental inference unit.
pub struct Cell {
    pub config: CellConfig,
    pub embedding: Array2<f32>,       // [vocab_size, d_model]
    pub output_head: Array2<f32>,     // [d_model, vocab_size]
    pub layers_tm: Vec<TimeMixLayer>,
    pub layers_ffn: Vec<SwiGLUFFN>,
}

impl Cell {
    /// Build a Cell from weights.
    /// Expected tensor names:
    ///   embedding.weight [vocab, d_model]
    ///   tm.{i}.wk.weight, tm.{i}.wv.weight, tm.{i}.wr.weight, tm.{i}.wo.weight [d_model, d_model]
    ///   tm.{i}.decay, tm.{i}.key_decay, tm.{i}.value_decay, tm.{i}.recept_decay [d_model]
    ///   tm.{i}.ln.weight, tm.{i}.ln.bias [d_model]
    ///   ffn.{i}.w1.weight [d_model, d_ffn], ffn.{i}.w2.weight [d_ffn, d_model], ffn.{i}.w3.weight [d_model, d_ffn]
    ///   ffn.{i}.ln.weight, ffn.{i}.ln.bias [d_model]
    ///   output.weight [d_model, vocab_size]
    pub fn from_weights(weights: CellWeights) -> Self {
        let config = weights.config.clone();
        let embedding = weights.get_2d("embedding.weight");
        let output_head = weights.get_2d("output.weight");

        let mut layers_tm = Vec::with_capacity(config.n_layers);
        let mut layers_ffn = Vec::with_capacity(config.n_layers);

        for i in 0..config.n_layers {
            layers_tm.push(TimeMixLayer::new(&weights, i));
            layers_ffn.push(SwiGLUFFN::new(&weights, i));
        }

        Self { config, embedding, output_head, layers_tm, layers_ffn }
    }

    /// Initialize random weights for a given config.
    /// Uses simple Xavier initialization.
    pub fn random(config: CellConfig) -> Self {
                let mut rng = rand::thread_rng();

        let mut weights = CellWeights::new(config.clone());

        // Embedding: [vocab, d_model]
        let scale = (1.0 / config.d_model as f32).sqrt();
        weights.set(
            "embedding.weight".into(),
            (0..config.vocab_size * config.d_model).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale).collect(),
            vec![config.vocab_size, config.d_model],
        );

        // Output head: [d_model, vocab_size]
        weights.set(
            "output.weight".into(),
            (0..config.d_model * config.vocab_size).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale).collect(),
            vec![config.d_model, config.vocab_size],
        );

        // Layers
        for i in 0..config.n_layers {
            let p = format!("tm.{}.", i);

            // time-mix projections: [d_model, d_model]
            let d2 = config.d_model * config.d_model;
            let scale2 = (2.0 / config.d_model as f32).sqrt();
            for name in &["wk.weight", "wv.weight", "wr.weight", "wo.weight"] {
                weights.set(
                    format!("{}{}", p, name),
                    (0..d2).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale2).collect(),
                    vec![config.d_model, config.d_model],
                );
            }

            // Decay/mixing parameters: [d_model], initialized to specific values
            // decay: negative values (so exp(-exp(x)) is small = fast decay)
            weights.set(
                format!("{}decay", p),
                (0..config.d_model).map(|_| rng.gen::<f32>() * 0.5 - 1.0).collect(),
                vec![config.d_model],
            );
            // key/value/receptance mixing: close to 0.5 = balanced between current and previous
            weights.set(
                format!("{}key_decay", p),
                (0..config.d_model).map(|_| 0.5 + rng.gen::<f32>() * 0.1 - 0.05).collect(),
                vec![config.d_model],
            );
            weights.set(
                format!("{}value_decay", p),
                (0..config.d_model).map(|_| 0.5 + rng.gen::<f32>() * 0.1 - 0.05).collect(),
                vec![config.d_model],
            );
            weights.set(
                format!("{}recept_decay", p),
                (0..config.d_model).map(|_| 0.5 + rng.gen::<f32>() * 0.1 - 0.05).collect(),
                vec![config.d_model],
            );

            // layernorm
            weights.set(format!("{}ln.weight", p), vec![1.0; config.d_model], vec![config.d_model]);
            weights.set(format!("{}ln.bias", p), vec![0.0; config.d_model], vec![config.d_model]);

            // FFN
            let fp = format!("ffn.{}.", i);
            let scale_ffn = (2.0 / (config.d_model as f32 + config.d_ffn as f32)).sqrt();

            // w1: [d_model, d_ffn]
            let d_x_ffn = config.d_model * config.d_ffn;
            weights.set(
                format!("{}w1.weight", fp),
                (0..d_x_ffn).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale_ffn).collect(),
                vec![config.d_model, config.d_ffn],
            );
            // w3: [d_model, d_ffn]
            weights.set(
                format!("{}w3.weight", fp),
                (0..d_x_ffn).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale_ffn).collect(),
                vec![config.d_model, config.d_ffn],
            );
            // w2: [d_ffn, d_model]
            weights.set(
                format!("{}w2.weight", fp),
                (0..d_x_ffn).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale_ffn).collect(),
                vec![config.d_ffn, config.d_model],
            );
            // layernorm
            weights.set(format!("{}ln.weight", fp), vec![1.0; config.d_model], vec![config.d_model]);
            weights.set(format!("{}ln.bias", fp), vec![0.0; config.d_model], vec![config.d_model]);
        }

        Self::from_weights(weights)
    }

    /// Generate token logits for a single token.
    pub fn forward_token(&self, token_id: usize, state: &mut CellState) -> Array1<f32> {
        
        // 1. Embedding lookup: [d_model]
        let mut x = self.embedding.row(token_id).to_owned();

        // 2. Through each layer: time-mix → ffn
        for i in 0..self.config.n_layers {
            x = self.layers_tm[i].forward(&x, &mut state.tm_states[i], state.step);
            x = self.layers_ffn[i].forward(&x);
        }

        state.step += 1;

        // 3. Output head: [vocab_size]
        self.output_head.t().dot(&x)
    }

    /// Generate next token (argmax sampling).
    pub fn generate_token(&self, token_id: usize, state: &mut CellState) -> usize {
        let logits = self.forward_token(token_id, state);
        argmax(&logits)
    }

    /// Generate text from a prompt.
    pub fn generate(&self, tokens: &[usize], max_new: usize) -> Vec<usize> {
        let mut state = CellState::new(&self.config);
        let mut output = tokens.to_vec();

        // Process prompt tokens
        for &t in tokens {
            let _ = self.forward_token(t, &mut state);
        }

        // Generate new tokens
        for _ in 0..max_new {
            let last = *output.last().unwrap();
            let next = self.generate_token(last, &mut state);
            output.push(next);

            // EOS detection (token 0)
            if next == 0 {
                break;
            }
        }

        output
    }

    /// Generate with temperature sampling.
    pub fn generate_with_sampling(
        &self, tokens: &[usize], max_new: usize, temperature: f32,
    ) -> Vec<usize> {
        let mut state = CellState::new(&self.config);
        let mut output = tokens.to_vec();

        for &t in tokens {
            let _ = self.forward_token(t, &mut state);
        }

        for _ in 0..max_new {
            let last = *output.last().unwrap();
            let logits = self.forward_token(last, &mut state);

            // Temperature scaling
            let scaled: Array1<f32> = if temperature > 0.0 {
                logits.mapv(|v| v / temperature)
            } else {
                logits
            };

            let probs = softmax(&scaled);
            let next = sample_categorical(&probs);
            output.push(next);

            if next == 0 { break; }
        }

        output
    }
}

// ===================== Cell State =====================

/// Runtime state for a Cell during generation.
/// Small and fixed-size — doesn't grow with sequence length.
#[derive(Debug, Clone)]
pub struct CellState {
    pub tm_states: Vec<TimeMixState>,
    pub step: usize,
}

impl CellState {
    pub fn new(config: &CellConfig) -> Self {
        Self {
            tm_states: (0..config.n_layers).map(|_| TimeMixState::new(config.n_heads)).collect(),
            step: 0,
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.tm_states {
            s.aa.fill(0.0);
            s.bb.fill(0.0);
            s.pp.fill(0.0);
        }
        self.step = 0;
    }
}

// ===================== Utility =====================

/// Argmax of a 1D array.
pub fn argmax(x: &Array1<f32>) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Sample from a probability distribution (categorical sampling).
pub fn sample_categorical(probs: &Array1<f32>) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1 // fallback to last
}

// ===================== Tests =====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_config_standard() {
        let cfg = CellConfig::standard();
        assert_eq!(cfg.vocab_size, 50277);
        assert_eq!(cfg.d_model, 768);
        assert!(cfg.total_params() > 100_000_000);
        assert!(cfg.total_params() < 300_000_000);
    }

    #[test]
    fn test_cell_config_tiny() {
        let cfg = CellConfig::tiny();
        assert_eq!(cfg.vocab_size, 1024);
        assert!(cfg.total_params() < 10_000_000);
    }

    #[test]
    fn test_layer_norm() {
        let x = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let out = layer_norm(&x, &w, 1e-5);
        // Mean should be ~0, values should be normalized
        let mean: f32 = out.sum() / out.len() as f32;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.0);
        assert!(silu(1.0) < 1.0);
        assert!(silu(-1.0) < 0.0);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&x);
        // Sum should be ~1
        let sum: f32 = s.sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Higher values should have higher probability
        assert!(s[2] > s[1]);
        assert!(s[1] > s[0]);
    }

    #[test]
    fn test_argmax() {
        let x = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.9, 0.2]);
        assert_eq!(argmax(&x), 3);
    }

    #[test]
    fn test_cell_random_forward() {
        let cfg = CellConfig::tiny();
        let cell = Cell::random(cfg);
        let mut state = CellState::new(&cell.config);

        // Should not panic
        let logits = cell.forward_token(42, &mut state);
        assert_eq!(logits.len(), cell.config.vocab_size);

        // All finite
        for &v in logits.iter() {
            assert!(v.is_finite(), "non-finite logit: {}", v);
        }
    }

    #[test]
    fn test_cell_generate() {
        let cfg = CellConfig::tiny();
        let cell = Cell::random(cfg);

        let output = cell.generate(&[1, 2, 3], 5);
        // Should have prompt + up to 5 new tokens
        assert!(output.len() >= 3);
        assert!(output.len() <= 8);
    }

    #[test]
    fn test_cell_state_reset() {
        let cfg = CellConfig::tiny();
        let mut state = CellState::new(&cfg);
        state.tm_states[0].aa[0] = 42.0;
        state.step = 10;
        state.reset();
        assert_eq!(state.tm_states[0].aa[0], 0.0);
        assert_eq!(state.step, 0);
    }

    #[test]
    fn test_weights_set_get() {
        let cfg = CellConfig::tiny();
        let mut w = CellWeights::new(cfg);
        w.set("test.1d".into(), vec![1.0, 2.0, 3.0], vec![3]);
        let arr = w.get_1d("test.1d");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[1], 2.0);
    }

    #[test]
    fn test_sample_categorical() {
        // All probability on index 2
        let probs = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        assert_eq!(sample_categorical(&probs), 2);
    }
}
