//! RWKV-4-169M model — loads binary + complete forward pass
//!
//! Binary format (little-endian, export_rwkv_weights.py):
//!   Header: vocab uint32, n_layers uint32, hidden uint32, ffn uint32, magic[4]
//!   emb: [V*H] float32
//!   Per layer: ln0_w/b [H] (layer 0 only), ln1_w/b, ln2_w/b [H],
//!     att_k/v/r/o [H*H], att_decay/first [H], att_mix_k/r/v [H],
//!     ffn_k/v/r [FFN*H or H*FFN], mix_k/r [H]
//!   Final: ln_out_w/b [H], head [V*H]

use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::io::Read;

// RWKV-4-169M constants
pub const VOCAB: usize = 50277;
pub const HIDDEN: usize = 768;
pub const LAYERS: usize = 12;
pub const FFN: usize = 3072;

// ===================== Binary Reading =====================

fn read_f32_into(r: &mut std::io::BufReader<std::fs::File>, n: usize) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    r.read_exact(&mut buf)?;
    let mut out = Vec::with_capacity(n);
    for chunk in buf.chunks(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn read_u32(r: &mut std::io::BufReader<std::fs::File>) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

// Convert ndarray row to Vec<f32>
fn row_to_vec(arr: &Array2<f32>, row: usize) -> Vec<f32> {
    arr.row(row).to_vec()
}

fn col_to_vec(arr: &Array2<f32>, col: usize) -> Vec<f32> {
    arr.column(col).to_vec()
}

// Gemv: [out_dim] = [out_dim, in_dim] @ [in_dim]
fn gemv(out: &mut [f32], w: &Array2<f32>, x: &[f32]) {
    for (oi, row) in w.rows().into_iter().enumerate() {
        out[oi] = row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
}

// Gemv that returns owned Vec<f32>
fn gemv_owned(w: &Array2<f32>, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w.nrows()];
    gemv(&mut out, w, x);
    out
}

// ===================== Model Loading =====================

#[derive(Clone)]
pub struct RwkvAttWeights {
    pub k_w: Array2<f32>,
    pub v_w: Array2<f32>,
    pub r_w: Array2<f32>,
    pub o_w: Array2<f32>,
    pub decay: Vec<f32>,
    pub first: Vec<f32>,
    pub mix_k: Vec<f32>,
    pub mix_r: Vec<f32>,
    pub mix_v: Vec<f32>,
}

#[derive(Clone)]
pub struct RwkvFfnWeights {
    pub k_w: Array2<f32>,
    pub v_w: Array2<f32>,
    pub r_w: Array2<f32>,
    pub mix_k: Vec<f32>,
    pub mix_r: Vec<f32>,
}

#[derive(Clone)]
pub struct RwkvLayerWeights {
    pub has_ln0: bool,
    pub ln0_w: Vec<f32>,
    pub ln0_b: Vec<f32>,
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub att: RwkvAttWeights,
    pub ffn: RwkvFfnWeights,
}

pub struct RwkvModel {
    pub emb: Array2<f32>,
    pub layers: Vec<RwkvLayerWeights>,
    pub ln_out_w: Vec<f32>,
    pub ln_out_b: Vec<f32>,
    pub head: Array2<f32>,
}

impl RwkvModel {
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::BufReader;
        use std::io::Read;

        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let vocab = read_u32(&mut r)? as usize;
        let n_layers = read_u32(&mut r)? as usize;
        let hidden = read_u32(&mut r)? as usize;
        let ffn_size = read_u32(&mut r)? as usize;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"RWKV", "Bad magic number");
        assert_eq!(vocab, VOCAB, "Vocab mismatch");
        assert_eq!(hidden, HIDDEN, "Hidden mismatch");

        // Embedding [V, H]
        let emb_data = read_f32_into(&mut r, VOCAB * HIDDEN)?;
        let emb = Array2::from_shape_vec((VOCAB, HIDDEN), emb_data).unwrap();

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let has_ln0 = i == 0;
            let ln0_w = if has_ln0 { read_f32_into(&mut r, HIDDEN)? } else { vec![0.0; HIDDEN] };
            let ln0_b = if has_ln0 { read_f32_into(&mut r, HIDDEN)? } else { vec![0.0; HIDDEN] };

            let ln1_w = read_f32_into(&mut r, HIDDEN)?;
            let ln1_b = read_f32_into(&mut r, HIDDEN)?;
            let ln2_w = read_f32_into(&mut r, HIDDEN)?;
            let ln2_b = read_f32_into(&mut r, HIDDEN)?;

            // Att: [H,H]
            let k_data = read_f32_into(&mut r, HIDDEN * HIDDEN)?;
            let v_data = read_f32_into(&mut r, HIDDEN * HIDDEN)?;
            let r_data = read_f32_into(&mut r, HIDDEN * HIDDEN)?;
            let o_data = read_f32_into(&mut r, HIDDEN * HIDDEN)?;
            let decay = read_f32_into(&mut r, HIDDEN)?;
            let first = read_f32_into(&mut r, HIDDEN)?;
            let mix_k = read_f32_into(&mut r, HIDDEN)?;
            let mix_r = read_f32_into(&mut r, HIDDEN)?;
            let mix_v = read_f32_into(&mut r, HIDDEN)?;

            // FFN: k_w [FFN,H], v_w [H,FFN], r_w [H,H]
            let ffn_k_data = read_f32_into(&mut r, FFN * HIDDEN)?;
            let ffn_v_data = read_f32_into(&mut r, HIDDEN * FFN)?;
            let ffn_r_data = read_f32_into(&mut r, HIDDEN * HIDDEN)?;
            let ffn_k = Array2::from_shape_vec((FFN, HIDDEN), ffn_k_data).unwrap();
            let ffn_v = Array2::from_shape_vec((HIDDEN, FFN), ffn_v_data).unwrap();
            let ffn_r_struct = Array2::from_shape_vec((HIDDEN, HIDDEN), ffn_r_data).unwrap();
            let ffn_mix_k = read_f32_into(&mut r, HIDDEN)?;
            let ffn_mix_r = read_f32_into(&mut r, HIDDEN)?;

            layers.push(RwkvLayerWeights {
                has_ln0,
                ln0_w, ln0_b,
                ln1_w, ln1_b, ln2_w, ln2_b,
                att: RwkvAttWeights {
                    k_w: Array2::from_shape_vec((HIDDEN, HIDDEN), k_data).unwrap(),
                    v_w: Array2::from_shape_vec((HIDDEN, HIDDEN), v_data).unwrap(),
                    r_w: Array2::from_shape_vec((HIDDEN, HIDDEN), r_data).unwrap(),
                    o_w: Array2::from_shape_vec((HIDDEN, HIDDEN), o_data).unwrap(),
                    decay, first, mix_k, mix_r, mix_v,
                },
                ffn: RwkvFfnWeights { k_w: ffn_k, v_w: ffn_v, r_w: ffn_r_struct, mix_k: ffn_mix_k, mix_r: ffn_mix_r },
            });
        }

        let ln_out_w = read_f32_into(&mut r, HIDDEN)?;
        let ln_out_b = read_f32_into(&mut r, HIDDEN)?;
        let head_data = read_f32_into(&mut r, VOCAB * HIDDEN)?;
        let head = Array2::from_shape_vec((VOCAB, HIDDEN), head_data).unwrap();

        Ok(RwkvModel { emb, layers, ln_out_w, ln_out_b, head })
    }

    pub fn total_params(&self) -> usize {
        let mut n = self.emb.len();
        for l in &self.layers {
            n += l.att.k_w.len() * 4; // k,v,r,o
            n += l.att.decay.len() * 2;
            n += l.att.mix_k.len() * 3;
            n += l.ffn.k_w.len() * 2;
            n += l.ffn.r_w.len();
            n += l.ffn.mix_k.len() * 2;
        }
        n += self.ln_out_w.len() * 2;
        n += self.head.len();
        n
    }
}

// ===================== Math =====================

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn gelu(x: f32) -> f32 {
    let c = std::f32::consts::PI.sqrt() * (2.0 / std::f32::consts::PI).sqrt();
    let x3 = x * x * x;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x3)).tanh())
}

fn layer_norm(x: &[f32], w: &[f32], b: f32, eps: f32) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let std = (var + eps).sqrt();
    x.iter().enumerate().map(|(i, xi)| (xi - mean) / std * (w[i] + 1.0) + b).collect()
}

// Weighted sum: out[i] = sum_j(w[i][j] * x[j])
fn weighted_sum(w: &Array2<f32>, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w.nrows()];
    for (oi, row) in w.rows().into_iter().enumerate() {
        out[oi] = row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
    out
}

// ===================== State =====================

#[derive(Clone)]
pub struct RwkvAttState {
    pub aa: Vec<f32>,    // [H] wkv accumulator numer
    pub bb: Vec<f32>,    // [H] wkv accumulator denom
    pub k_prev: Vec<f32>,
    pub r_prev: Vec<f32>,
    pub v_prev: Vec<f32>,
}

#[derive(Clone)]
pub struct RwkvFfnState {
    pub k_prev: Vec<f32>,
    pub r_prev: Vec<f32>,
}

#[derive(Clone)]
pub struct RwkvLayerState {
    pub att: RwkvAttState,
    pub ffn: RwkvFfnState,
}

#[derive(Clone)]
pub struct RwkvModelState {
    pub layers: Vec<RwkvLayerState>,
}

impl RwkvModelState {
    pub fn new() -> Self {
        let layer = || RwkvLayerState {
            att: RwkvAttState { aa: vec![0.0; HIDDEN], bb: vec![0.0; HIDDEN],
                k_prev: vec![0.0; HIDDEN], r_prev: vec![0.0; HIDDEN], v_prev: vec![0.0; HIDDEN] },
            ffn: RwkvFfnState { k_prev: vec![0.0; HIDDEN], r_prev: vec![0.0; HIDDEN] },
        };
        RwkvModelState { layers: (0..LAYERS).map(|_| layer()).collect() }
    }
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.att.aa.fill(0.0); l.att.bb.fill(0.0);
            l.att.k_prev.fill(0.0); l.att.r_prev.fill(0.0); l.att.v_prev.fill(0.0);
            l.ffn.k_prev.fill(0.0); l.ffn.r_prev.fill(0.0);
        }
    }
}

// ===================== Forward =====================

impl RwkvModel {
    /// Forward one token ID → logits over vocab
    pub fn forward_token(&self, token_id: usize, state: &mut RwkvModelState) -> Vec<f32> {
        // 1. Embedding
        let mut x: Vec<f32> = self.emb.row(token_id).to_vec();

        // 2. Blocks
        for (i, layer) in self.layers.iter().enumerate() {
            let s = &mut state.layers[i];

            // Pre-norm 1
            let xn = layer_norm(&x, &layer.ln1_w, layer.ln1_b[0], 1e-5);

            // Attention time mix
            let k_mix: Vec<f32> = (0..HIDDEN).map(|j| layer.att.mix_k[j] * xn[j] + (1.0 - layer.att.mix_k[j]) * s.att.k_prev[j]).collect();
            let r_mix: Vec<f32> = (0..HIDDEN).map(|j| layer.att.mix_r[j] * xn[j] + (1.0 - layer.att.mix_r[j]) * s.att.r_prev[j]).collect();
            let v_mix: Vec<f32> = (0..HIDDEN).map(|j| layer.att.mix_v[j] * xn[j] + (1.0 - layer.att.mix_v[j]) * s.att.v_prev[j]).collect();

            // Project k, r, v
            let k = weighted_sum(&layer.att.k_w, &k_mix); // [H]
            let r = weighted_sum(&layer.att.r_w, &r_mix); // [H]
            let v_in = weighted_sum(&layer.att.v_w, &v_mix); // [H]

            // RWKV-4 WKV (per-channel recurrence)
            let mut wkv = vec![0.0f32; HIDDEN];
            let mut new_aa = vec![0.0f32; HIDDEN];
            let mut new_bb = vec![0.0f32; HIDDEN];
            for j in 0..HIDDEN {
                let dec = layer.att.decay[j].exp();
                new_aa[j] = dec * s.att.aa[j] + v_in[j] * k[j];
                new_bb[j] = dec * s.att.bb[j] + k[j];
                wkv[j] = if new_bb[j].abs() > 1e-7 { new_aa[j] / new_bb[j] } else { layer.att.first[j] };
            }

            // Update state
            for j in 0..HIDDEN {
                s.att.aa[j] = new_aa[j];
                s.att.bb[j] = new_bb[j];
                let dec = layer.att.decay[j].exp();
                s.att.k_prev[j] = dec * s.att.k_prev[j] + k_mix[j];
                s.att.r_prev[j] = dec * s.att.r_prev[j] + r_mix[j];
                s.att.v_prev[j] = dec * s.att.v_prev[j] + v_mix[j];
            }

            // r * wkv
            let rwkv: Vec<f32> = wkv.iter().zip(r.iter()).map(|(wkv_j, r_j)| sigmoid(*r_j) * wkv_j).collect();

            // Output projection
            let att_out = weighted_sum(&layer.att.o_w, &rwkv);

            // Residual
            for j in 0..HIDDEN { x[j] += att_out[j]; }

            // Pre-norm 2
            let xn2 = layer_norm(&x, &layer.ln2_w, layer.ln2_b[0], 1e-5);

            // FFN time mix
            let ffn_k_mix: Vec<f32> = (0..HIDDEN).map(|j| layer.ffn.mix_k[j] * xn2[j] + (1.0 - layer.ffn.mix_k[j]) * s.ffn.k_prev[j]).collect();
            let ffn_r_mix: Vec<f32> = (0..HIDDEN).map(|j| layer.ffn.mix_r[j] * xn2[j] + (1.0 - layer.ffn.mix_r[j]) * s.ffn.r_prev[j]).collect();

            // FFN: k = FFN[H,H] @ x, v = H[H,FFN] @ FFN_out, r = H[H,H] @ x
            let ffn_k = weighted_sum(&layer.ffn.k_w, &ffn_k_mix); // [FFN]
            let ffn_r = weighted_sum(&layer.ffn.r_w, &ffn_r_mix); // [H]
            let ffn_v: Vec<f32> = {
                let mut out = vec![0.0f32; HIDDEN];
                for h in 0..HIDDEN {
                    out[h] = layer.ffn.v_w.row(h).iter().zip(ffn_k.iter()).map(|(wi, ki)| wi * ki).sum();
                }
                out
            };

            // Update state
            for j in 0..HIDDEN {
                let dec = layer.ffn.mix_k[j].exp().min(1.0);
                s.ffn.k_prev[j] = dec * s.ffn.k_prev[j] + ffn_k_mix[j];
                s.ffn.r_prev[j] = dec * s.ffn.r_prev[j] + ffn_r_mix[j];
            }

            // FFN output: r * GELU(k)
            let ffn_out: Vec<f32> = (0..HIDDEN).map(|j| sigmoid(ffn_r[j]) * gelu(ffn_v[j])).collect();

            // Residual
            for j in 0..HIDDEN { x[j] += ffn_out[j]; }
        }

        // 3. Final norm
        x = layer_norm(&x, &self.ln_out_w, self.ln_out_b[0], 1e-5);

        // 4. Output projection
        let logits = weighted_sum(&self.head, &x);
        logits
    }

    /// Generate tokens autoregressively
    pub fn generate(&self, input_ids: &[usize], max_new: usize, temperature: f32) -> (Vec<usize>, std::time::Duration) {
        use std::time::Instant;
        let start = Instant::now();

        let mut state = RwkvModelState::new();
        let mut all_ids = input_ids.to_vec();

        // Forward prompt tokens
        for &id in input_ids {
            self.forward_token(id, &mut state);
        }

        // Generate
        for _ in 0..max_new {
            let logits = self.forward_token(*all_ids.last().unwrap(), &mut state);
            let next = sample_token(&logits, temperature);
            all_ids.push(next);
            if next == 0 { break; }
        }

        (all_ids, start.elapsed())
    }
}

/// Sample next token from logits
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    if temperature < 1e-6 {
        return logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }

    // Temperature
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();

    // Softmax
    let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_s: Vec<f32> = scaled.iter().map(|v| (v - max_s).exp()).collect();
    let sum_exp = exp_s.iter().sum::<f32>();
    let probs: Vec<f32> = exp_s.iter().map(|v| v / sum_exp).collect();

    // Top-p sampling
    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, p)| (i, *p)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let threshold = 0.9_f32;
    let mut cumsum = 0.0_f32;
    let mut top: Vec<(usize, f32)> = Vec::new();
    for p in &pairs {
        if cumsum >= threshold { break; }
        cumsum += p.1;
        top.push(*p);
    }

    // Renormalize
    let total: f32 = top.iter().map(|(_, p)| p).sum();
    let r: f32 = rand::thread_rng().gen();
    let mut acc = 0.0_f32;
    for (i, p) in top {
        acc += p / total;
        if r <= acc { return i; }
    }
    pairs.last().map(|(i, _)| *i).unwrap_or(0)
}
