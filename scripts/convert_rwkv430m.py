"""Convert RWKV-4-World-430M HF weights to native format"""
import sys
sys.path.insert(0, r'D:\pylib')
import torch

print('Loading HF weights...')
w = torch.load(r'D:\pylib\pytorch_model.bin', map_location='cpu', weights_only=True)

new_w = {}
for k, v in w.items():
    if k == 'head.weight':
        new_w['head.weight'] = v
        continue
    if k.startswith('rwkv.'):
        k = k[5:]
    k = k.replace('embeddings.weight', 'emb.weight')
    k = k.replace('.attention.', '.att.')
    k = k.replace('.feed_forward.', '.ffn.')
    k = k.replace('.pre_ln.', '.ln0.')
    k = k.replace('time_mix_key', 'time_mix_k')
    k = k.replace('time_mix_value', 'time_mix_v')
    k = k.replace('time_mix_receptance', 'time_mix_r')
    new_w[k] = v

print(f'Converted {len(new_w)} keys')
# Check critical keys
for needed in ['emb.weight', 'blocks.0.ln0.weight', 'blocks.0.att.time_mix_k', 'head.weight']:
    if needed in new_w:
        print(f'  {needed}: {new_w[needed].shape} {new_w[needed].dtype}')
    else:
        print(f'  MISSING: {needed}')

out_path = r'D:\pylib\rwkv-4-world-430m-native.pth'
torch.save(new_w, out_path)
import os
print(f'Saved: {os.path.getsize(out_path)//1024//1024}MB')
