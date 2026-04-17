import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'D:\pylib')

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import json

print('[1] Loading model...', flush=True)
m = RWKV(model=r'D:\pylib\rwkv-4-169m-native.pth', strategy='cpu fp32')
print('[2] Model loaded', flush=True)

print('[3] Loading tokenizer...', flush=True)
t = TRIE_TOKENIZER(r'D:\pylib\rwkv4-world-tok\rwkv_vocab_v20230424.txt')
print('[4] Tokenizer loaded', flush=True)

# Try encoding first
prompt = 'def hello():'
try:
    tokens = t.encode(prompt)
    print(f'[5] Encoded prompt: {len(tokens)} tokens', flush=True)
except Exception as e:
    print(f'[5] Encode error: {e}', flush=True)
    # Fallback to simple tokenization
    tokens = [ord(c) % 50277 for c in prompt]
    print(f'[5] Fallback: {len(tokens)} tokens', flush=True)

# Just do one forward pass, no generation loop
state = None
out, state = m.forward(tokens[-1:], state)
print(f'[6] Forward OK, shape: {out.shape}', flush=True)
print(f'    Top 5 tokens: {out.topk(5).indices.tolist()}', flush=True)
print('Done', flush=True)