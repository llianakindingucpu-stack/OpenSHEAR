import sys, time
sys.path.insert(0, 'D:/IdeaProjects/decentral-ai/scripts')
sys.path.insert(0, 'D:/pylib')

from rwkv_engine import load_model, RWKVTokenizer

print('Loading model...')
t0 = time.time()
model, cfg = load_model('D:/pylib/rwkv-4-169m-native.pth')
tok = RWKVTokenizer()
vocab = cfg.get('vocab', 65536)
print(f'Loaded in {time.time()-t0:.1f}s, vocab={vocab}, layers={cfg.get("n_layers")}')

prompt = 'Say hello in exactly 3 words'
tokens = tok.encode(prompt)
print(f'Prompt: "{prompt}"')
print(f'Raw tokens: {tokens}')
print(f'Max token: {max(tokens) if tokens else 0}, vocab={vocab}')

tokens_clip = [t for t in tokens if 0 <= t < vocab]
print(f'Clipped tokens: {tokens_clip}')
print(f'Tokens clipped: {len(tokens) - len(tokens_clip)} tokens removed')

import torch
t = torch.tensor([tokens_clip], dtype=torch.long)
print(f'Input tensor: shape={t.shape}, range=[{t.min().item()}, {t.max().item()}]')
logits, state = model(t, None)
print(f'Forward OK, logits shape={logits.shape}, range=[{logits.min().item():.2f}, {logits.max().item():.2f}]')

# Sample one token
lp = logits[0, -1, :].float()
top_val, top_idx = torch.topk(lp, 5)
print(f'Top 5 logits: val={top_val.tolist()}, idx={top_idx.tolist()}')
print(f'Vocab size in logits: {logits.shape[-1]}')
