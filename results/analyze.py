import json

with open(r'D:\IdeaProjects\decentral-ai\results\humaneval_rust_full.jsonl') as f:
    results = [json.loads(l) for l in f if l.strip()]

print(f"Total results: {len(results)}\n")

for r in results:
    tid = r['task_id']
    entry = r['entry_point']
    comp = r.get('completion_text', '')
    full = r.get('full_code', '')
    tokens = r.get('completion_tokens', 0)
    speed = r.get('speed_tok_per_s', 0)
    prompt_tokens = r.get('prompt_tokens', 0)

    has_func = ('def ' + entry) in full
    has_return = 'return' in comp
    lines = comp.split('\n')
    ret_lines = [l.strip() for l in lines if 'return' in l and not l.strip().startswith('"""') and not l.strip().startswith("'''") and not l.strip().startswith('#')]

    # Check if it looks like it actually completed the function
    ends_properly = any('return' in l for l in lines[-5:])
    has_colon_end = any(l.strip().startswith(':') for l in lines[-3:])

    print(f"=== {tid} ({entry}) ===")
    print(f"  func defined: {has_func}")
    print(f"  has return:   {has_return}")
    print(f"  returns:      {ret_lines[:3]}")
    print(f"  tokens:       {tokens} ({speed:.1f} tok/s)")
    print(f"  prompt:       {prompt_tokens}")
    print(f"  completion chars: {len(comp)}")
    print()
