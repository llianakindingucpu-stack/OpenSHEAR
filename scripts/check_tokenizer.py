import json, sys

with open(r'D:/pylib/tokenizer.json','r',encoding='utf-8') as f:
    d=json.load(f)

vocab = d['model']['vocab']

# Check individual tokens
for w in ['hello', 'Hello', 'The', 'the', ' T', 'Th', 'he', 'el', 'll', 'lo', 'Ġ', 'ĠT', 'ĠThe', 'Ġthe']:
    if w in vocab:
        print(f'  "{w}" -> id={vocab[w]}')
    else:
        print(f'  "{w}" -> NOT FOUND')

# Check pre_tokenizer
pt = d.get('pre_tokenizer', {})
print(f'\npre_tokenizer type: {pt.get("type")}')
if 'pre_tokenizer' in pt:
    print(f'  inner: {pt["pre_tokenizer"].get("type")}')
print(f'decoder type: {d.get("decoder",{}).get("type")}')

# Try to understand the byte-level encoding
# GPT-NeoX uses Ġ prefix for space-prefixed tokens
g_tokens = [k for k in vocab.keys() if k.startswith('\u0120')]
print(f'\nTokens starting with G-with-dot: {len(g_tokens)}')
print('Samples:', g_tokens[:10])

# Try actual encode with tokenizers library if available
try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(r'D:/pylib/tokenizer.json')
    enc = tok.encode('The quick brown fox')
    print(f'\nActual encode("The quick brown fox"): {enc.ids}')
    print(f'Tokens: {[tok.decode([i]) for i in enc.ids]}')
except ImportError:
    print('\ntokenizers library not available, using manual check')
    # Byte-level: space = 0x20 -> Ġ (0x120)
    # 'The' with leading space = ' The' -> bytes [0x20, 0x54, 0x68, 0x65]
    # In GPT-NeoX BPE: 'ĠThe' represents ' The'
    text = 'The quick brown fox'
    # Byte-level encode: each byte -> unicode char
    byte_encoded = ''.join(chr(b) if b >= 33 else chr(b + 256) for b in text.encode('utf-8'))
    print(f'Byte-encoded: {repr(byte_encoded)}')
