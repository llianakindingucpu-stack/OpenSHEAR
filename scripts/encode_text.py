import json, sys, os

tok_path = r'D:/pylib/tokenizer.json'

try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    text = sys.argv[1] if len(sys.argv) > 1 else 'The quick brown fox'
    enc = tok.encode(text)
    # Output: space-separated IDs
    print(' '.join(str(i) for i in enc.ids))
except ImportError:
    # Fallback: manual byte-level BPE
    with open(tok_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    vocab = d['model']['vocab']
    merges = d['model'].get('merges', [])

    text = sys.argv[1] if len(sys.argv) > 1 else 'The quick brown fox'

    # Byte-level encode: space (0x20) -> chr(0x120), etc.
    def byte_encode(s):
        return ''.join(chr(b) if b >= 33 else chr(b + 256) for b in s.encode('utf-8'))

    # BPE merge
    tokens = list(byte_encode(text))
    for a, b in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens[i:i+2] = [a + b]
            else:
                i += 1

    ids = [vocab.get(t, 0) for t in tokens]
    print(' '.join(str(i) for i in ids))
