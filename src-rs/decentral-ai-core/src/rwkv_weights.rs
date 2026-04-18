//! RWKV-4 native format Cell weights loaded from safetensors
//! 
//! RWKV-4-169M structure:
//!   emb.weight:        [50277, 768]
//!   blocks.{i}.ln0.weight:  [768]  (only block 0 has ln0, it's a pre-norm)
//!   blocks.{i}.ln1.weight/bias: [768]
//!   blocks.{i}.ln2.weight/bias: [768]
//!   blocks.{i}.att.key.weight:      [768, 768]
//!   blocks.{i}.att.value.weight:   [768, 768]
//!   blocks.{i}.att.receptance.weight: [768, 768]
//!   blocks.{i}.att.output.weight:  [768, 768]
//!   blocks.{i}.att.time_decay:      [768]
//!   blocks.{i}.att.time_first:     [768]
//!   blocks.{i}.att.time_mix_k:      [1, 1, 768]
//!   blocks.{i}.att.time_mix_r:      [1, 1, 768]
//!   blocks.{i}.att.time_mix_v:      [1, 1, 768]
//!   blocks.{i}.ffn.key.weight:      [3072, 768]
//!   blocks.{i}.ffn.value.weight:    [768, 3072]
//!   blocks.{i}.ffn.receptance.weight: [768, 768]
//!   blocks.{i}.ffn.time_mix_k:      [1, 1, 768]
//!   blocks.{i}.ffn.time_mix_r:      [1, 1, 768]
//!   ln_out.weight/bias:         [768]
//!   head.weight:                [50277, 768]

use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

// ===================== Weight Loading =====================

/// RWKV-4-169M has 12 layers, 768 hidden, 50277 vocab
pub const RWKV_VOCAB: usize = 50277;
pub const RWKV_HIDDEN: usize = 768;
pub const RWKV_LAYERS: usize = 12;
pub const RWKV_FFN: usize = 3072;

/// All weights for one RWKV layer
#[derive(Clone)]
pub struct RwkvLayerWeights {
    // Pre-norm (block 0 only has ln0; blocks 1+ start with ln1)
    pub ln0_w: Option<Array1<f32>>,  // [768] or None
    pub ln0_b: Option<Array1<f32>>,  // [768] or None

    // Layer norms
    pub ln1_w: Array1<f32>,  // [768]
    pub ln1_b: Array1<f32>,  // [768]
    pub ln2_w: Array1<f32>,  // [768]
    pub ln2_b: Array1<f32>,  // [768]

    // Attention
    pub att_k: Array2<f32>,   // [768, 768]
    pub att_v: Array2<f32>,   // [768, 768]
    pub att_r: Array2<f32>,   // [768, 768]
    pub att_o: Array2<f32>,   // [768, 768]
    pub att_decay: Array1<f32>, // [768]
    pub att_first: Array1<f32>, // [768]
    pub att_mix_k: Array3<f32>, // [1, 1, 768]
    pub att_mix_r: Array3<f32>, // [1, 1, 768]
    pub att_mix_v: Array3<f32>, // [1, 1, 768]

    // FFN
    pub ffn_k: Array2<f32>,    // [3072, 768]
    pub ffn_v: Array2<f32>,    // [768, 3072]
    pub ffn_r: Array2<f32>,    // [768, 768]
    pub ffn_mix_k: Array3<f32>, // [1, 1, 768]
    pub ffn_mix_r: Array3<f32>, // [1, 1, 768]
}

/// RWKV model with all layers loaded
#[derive(Clone)]
pub struct RwkvModelWeights {
    pub emb: Array2<f32>,       // [50277, 768]
    pub layers: Vec<RwkvLayerWeights>,
    pub ln_out_w: Array1<f32>,  // [768]
    pub ln_out_b: Array1<f32>,  // [768]
    pub head: Array2<f32>,      // [50277, 768]
}

/// Load weights from safetensors file, convert PyTorch format to ndarray.
/// PyTorch [M, N] → ndarray [N, M] (row-major: out_dim × in_dim)
pub fn load_from_safetensors(path: &str) -> std::io::Result<RwkvModelWeights> {
    // Use Python to load safetensors, then we process in Rust
    // (No Python available in Rust, so we do this via subprocess)
    // Actually, we do it inline using the safetensors_rs crate.
    // But for now, use Python subprocess to load + print as bincode,
    // then load bincode in Rust.
    //
    // Better approach: write a Python script that loads the weights
    // and serializes to a simple binary format (bincode), then load in Rust.
    unimplemented!("Use load_from_bincode() instead after running export script")
}

/// Simple binary format: each tensor is [u32 dims, u32*shape, f32*data] little-endian
pub fn load_from_bincode(path: &str) -> std::io::Result<RwkvModelWeights> {
    use std::fs::File;
    use std::io::{BufReader, Read};
    use byteorder::{LittleEndian, ReadBytesExt};

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    fn read_array1(r: &mut BufReader<File>) -> std::io::Result<Array1<f32>> {
        let len = r.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0.0f32; len];
        r.read_f32_into::<LittleEndian>(&mut data)?;
        Ok(Array1::from_vec(data))
    }
    fn read_array2(r: &mut BufReader<File>) -> std::io::Result<Array2<f32>> {
        let rows = r.read_u32::<LittleEndian>()? as usize;
        let cols = r.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0.0f32; rows * cols];
        r.read_f32_into::<LittleEndian>(&mut data)?;
        // Shape [rows, cols] → row-major [rows, cols]
        Ok(Array2::from_shape_vec((rows, cols), data).unwrap())
    }
    fn read_array3(r: &mut BufReader<File>) -> std::io::Result<Array3<f32>> {
        let d1 = r.read_u32::<LittleEndian>()? as usize;
        let d2 = r.read_u32::<LittleEndian>()? as usize;
        let d3 = r.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0.0f32; d1 * d2 * d3];
        r.read_f32_into::<LittleEndian>(&mut data)?;
        Ok(Array3::from_shape_vec((d1, d2, d3), data).unwrap())
    }

    let emb = read_array2(&mut reader)?;
    let n_layers = RWKV_LAYERS;
    let mut layers = Vec::with_capacity(n_layers);

    for i in 0..n_layers {
        let ln0_w = if i == 0 { Some(read_array1(&mut reader)?) } else { None };
        let ln0_b = if i == 0 { Some(read_array1(&mut reader)?) } else { None };
        let ln1_w = read_array1(&mut reader)?;
        let ln1_b = read_array1(&mut reader)?;
        let ln2_w = read_array1(&mut reader)?;
        let ln2_b = read_array1(&mut reader)?;
        let att_k = read_array2(&mut reader)?;
        let att_v = read_array2(&mut reader)?;
        let att_r = read_array2(&mut reader)?;
        let att_o = read_array2(&mut reader)?;
        let att_decay = read_array1(&mut reader)?;
        let att_first = read_array1(&mut reader)?;
        let att_mix_k = read_array3(&mut reader)?;
        let att_mix_r = read_array3(&mut reader)?;
        let att_mix_v = read_array3(&mut reader)?;
        let ffn_k = read_array2(&mut reader)?;
        let ffn_v = read_array2(&mut reader)?;
        let ffn_r = read_array2(&mut reader)?;
        let ffn_mix_k = read_array3(&mut reader)?;
        let ffn_mix_r = read_array3(&mut reader)?;

        layers.push(RwkvLayerWeights {
            ln0_w, ln0_b, ln1_w, ln1_b, ln2_w, ln2_b,
            att_k, att_v, att_r, att_o, att_decay, att_first,
            att_mix_k, att_mix_r, att_mix_v,
            ffn_k, ffn_v, ffn_r, ffn_mix_k, ffn_mix_r,
        });
    }

    let ln_out_w = read_array1(&mut reader)?;
    let ln_out_b = read_array1(&mut reader)?;
    let head = read_array2(&mut reader)?;

    Ok(RwkvModelWeights { emb, layers, ln_out_w, ln_out_b, head })
}
