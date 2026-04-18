#!/usr/bin/env python3
"""
Export RWKV-4-169M weights from safetensors to simple binary format for Rust.
Binary layout (little-endian):
  - uint32: vocab_size
  - uint32: n_layers  
  - uint32: hidden_size
  - uint32: ffn_size
  - uint32: magic "RWKV"
  - For each tensor: uint32 ndim, uint32[n] shape, float32[n] data
"""

import struct
import sys
from pathlib import Path

import safetensors.torch as st

def write_array1(f, arr):
    """Write 1D array: uint32 len, float32[len]"""
    arr = arr.flatten().float().numpy()
    f.write(struct.pack('<I', len(arr)))
    f.write(struct.pack(f'<{len(arr)}f', *arr.tolist()))

def write_array2(f, arr):
    """Write 2D array: uint32 rows, uint32 cols, float32[rows*cols]"""
    arr = arr.float().numpy()
    rows, cols = arr.shape
    f.write(struct.pack('<II', rows, cols))
    # Flatten row-major: [rows, cols] → [rows*cols]
    f.write(struct.pack(f'<{rows*cols}f', *arr.flatten().tolist()))

def write_array3(f, arr):
    """Write 3D array: uint32 d1, uint32 d2, uint32 d3, float32[d1*d2*d3]"""
    arr = arr.float().numpy()
    d1, d2, d3 = arr.shape
    f.write(struct.pack('<III', d1, d2, d3))
    # Flatten row-major: [d1, d2, d3] → [d1*d2*d3]
    f.write(struct.pack(f'<{d1*d2*d3}f', *arr.flatten().tolist()))

def export(model_path: str, output_path: str):
    print(f"Loading {model_path}...")
    tensors = st.load_file(model_path)
    keys = sorted(tensors.keys())
    print(f"Loaded {len(keys)} tensors")

    # Infer config from first layer
    hidden = tensors['emb.weight'].shape[1]  # [vocab, hidden]
    vocab = tensors['emb.weight'].shape[0]
    # Count unique layer numbers
    layer_nums = set()
    for k in keys:
        if k.startswith('blocks.'):
            parts = k.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_nums.add(int(parts[1]))
    n_layers = len(layer_nums)  # actual number of layers
    ffn = tensors['blocks.0.ffn.key.weight'].shape[0]  # [ffn, hidden]

    print(f"Config: vocab={vocab}, hidden={hidden}, layers={n_layers}, ffn={ffn}")

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', vocab))
        f.write(struct.pack('<I', n_layers))
        f.write(struct.pack('<I', hidden))
        f.write(struct.pack('<I', ffn))
        f.write(b'RWKV')  # magic

        # Embedding
        write_array2(f, tensors['emb.weight'])
        print(f"  emb: {tensors['emb.weight'].shape}")

        # Layers
        for i in range(n_layers):
            prefix = f'blocks.{i}'
            
            # Layer 0 has ln0; others don't
            if i == 0:
                write_array1(f, tensors[f'{prefix}.ln0.weight'])
                write_array1(f, tensors[f'{prefix}.ln0.bias'])
            else:
                # Dummy zeros for ln0 (Rust will skip)
                write_array1(f, tensors['ln_out.weight'] * 0)
                write_array1(f, tensors['ln_out.bias'] * 0)
            
            # ln1, ln2
            write_array1(f, tensors[f'{prefix}.ln1.weight'])
            write_array1(f, tensors[f'{prefix}.ln1.bias'])
            write_array1(f, tensors[f'{prefix}.ln2.weight'])
            write_array1(f, tensors[f'{prefix}.ln2.bias'])

            # Attention
            write_array2(f, tensors[f'{prefix}.att.key.weight'])         # [H, H]
            write_array2(f, tensors[f'{prefix}.att.value.weight'])       # [H, H]
            write_array2(f, tensors[f'{prefix}.att.receptance.weight']) # [H, H]
            write_array2(f, tensors[f'{prefix}.att.output.weight'])     # [H, H]
            write_array1(f, tensors[f'{prefix}.att.time_decay'])          # [H]
            write_array1(f, tensors[f'{prefix}.att.time_first'])         # [H]
            write_array3(f, tensors[f'{prefix}.att.time_mix_k'])         # [1,1,H]
            write_array3(f, tensors[f'{prefix}.att.time_mix_r'])         # [1,1,H]
            write_array3(f, tensors[f'{prefix}.att.time_mix_v'])         # [1,1,H]

            # FFN
            write_array2(f, tensors[f'{prefix}.ffn.key.weight'])          # [FFN, H]
            write_array2(f, tensors[f'{prefix}.ffn.value.weight'])       # [H, FFN]
            write_array2(f, tensors[f'{prefix}.ffn.receptance.weight'])  # [H, H]
            write_array3(f, tensors[f'{prefix}.ffn.time_mix_k'])          # [1,1,H]
            write_array3(f, tensors[f'{prefix}.ffn.time_mix_r'])          # [1,1,H]

            print(f"  layer {i} done")

        # Final norm
        write_array1(f, tensors['ln_out.weight'])
        write_array1(f, tensors['ln_out.bias'])

        # Head
        write_array2(f, tensors['head.weight'])

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/pylib/rwkv-4-169m.safetensors'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'D:/IdeaProjects/decentral-ai/data/rwkv4_169m.bin'
    export(model_path, output_path)
