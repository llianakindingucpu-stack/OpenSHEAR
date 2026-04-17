import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'D:\pylib')

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

print('[1] Loading model...', flush=True)
m = RWKV(model=r'D:\pylib\rwkv-4-169m-native.pth', strategy='cpu fp32')
print('[2] Model loaded', flush=True)

print('[3] Loading tokenizer...', flush=True)
t = TRIE_TOKENIZER(r'D:\pylib\rwkv4-world-tok\rwkv_vocab_v20230424.txt')
print('[4] Tokenizer loaded', flush=True)

# Generate
prompt = 'def hello():'
tokens = t.encode(prompt)
state = None

print(f'[5] Generating 30 tokens...', flush=True)
for i in range(30):
    out, state = m.forward(tokens[-1:], state)
    next_token = out.argmax().item()
    tokens.append(next_token)
    if next_token == 0:
        break

result = t.decode(tokens)
print(f'[6] Generated: {result[:150]}', flush=True)
print('Done', flush=True)