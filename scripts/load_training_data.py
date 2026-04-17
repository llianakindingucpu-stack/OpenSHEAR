"""Unified training data loader for DecentralAI evolution pipeline.

Supports:
  - HumanEval / CodexEval (OpenAI format)
  - synthetic_variations (generated variants)
  - APPS (competition format)
  - CodeAlpaca / Magicoder (instruction format)

Each dataset is converted to a unified JSONL format suitable for LoRA fine-tuning.
"""
import sys, os, json, random

DATA = r'D:\IdeaProjects\decentral-ai\data'
OUT  = os.path.join(DATA, 'unified_training.jsonl')

# ── Unified record schema ────────────────────────────────────────────────
UNIFIED_SCHEMA = {
    'id':          str,    # unique identifier
    'source':      str,    # dataset name
    'prompt':      str,    # instruction / problem statement
    'solution':    str,    # canonical solution code
    'test_code':  str,    # test harness (optional)
    'entry_point':str,    # function name (optional)
    'difficulty': str,    # easy / medium / hard (optional)
    'language':   str,    # python / javascript / ...
    'tags':        list,  # skill tags
}

# ── HumanEval format ─────────────────────────────────────────────────────
def load_humaneval(path):
    """task_id, prompt, canonical_solution, test, entry_point"""
    records = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line.decode('utf-8', 'replace'))
            except Exception:
                continue
            records.append({
                'id':          obj.get('task_id', obj.get('instance_id', '')),
                'source':      'humaneval',
                'prompt':      obj.get('prompt', ''),
                'solution':    obj.get('canonical_solution', obj.get('solution', '')),
                'test_code':   obj.get('test', ''),
                'entry_point': obj.get('entry_point', ''),
                'difficulty':  obj.get('difficulty', 'medium'),
                'language':   'python',
                'tags':        ['code-generation'],
            })
    return records

def load_synthetic(path):
    """Variation records with parent_id / is_variation metadata."""
    records = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line.decode('utf-8', 'replace'))
            except Exception:
                continue
            records.append({
                'id':          obj.get('variation_id', obj.get('task_id', '')),
                'source':      'synthetic_variations',
                'prompt':      obj.get('prompt', ''),
                'solution':    obj.get('canonical_solution', ''),
                'test_code':   obj.get('test', ''),
                'entry_point': obj.get('entry_point', ''),
                'difficulty':  obj.get('difficulty', 'medium'),
                'language':   'python',
                'tags':        ['code-generation', 'synthetic', 'variant'],
            })
    return records

# ── APPS format ──────────────────────────────────────────────────────────
def load_apps(path):
    """id, question, solutions, input_output, difficulty, starter_code"""
    records = []
    with open(path, 'rb') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line.decode('utf-8', 'replace'))
            except Exception:
                continue
            sols_raw = obj.get('solutions', '')
            # Handle both list and string (JSON string of list)
            if isinstance(sols_raw, str):
                try:
                    sols = json.loads(sols_raw)
                except:
                    sols = [sols_raw]
            else:
                sols = sols_raw if isinstance(sols_raw, list) else [sols_raw]
            # Take first solution
            sol = sols[0] if sols else ''
            if isinstance(sol, list):
                sol = '\n'.join(sol)
            # Skip if solution is too long (>2048 chars for 169M model)
            if len(sol) > 2048:
                sol = sol[:2048]
            records.append({
                'id':          f"apps_{obj.get('id', i)}",
                'source':      'apps',
                'prompt':      obj.get('question', ''),
                'solution':    sol,
                'test_code':   '',
                'entry_point': '',
                'difficulty':  obj.get('difficulty', 'medium'),
                'language':   'python',
                'tags':        ['code-generation', 'competitive', obj.get('difficulty','')],
            })
    return records

# ── Instruction format (CodeAlpaca / Magicoder) ─────────────────────────
def load_instruction(path, source_name):
    """instruction/response format."""
    records = []
    with open(path, 'rb') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line.decode('utf-8', 'replace'))
            except Exception:
                continue
            records.append({
                'id':          f"{source_name}_{i}",
                'source':      source_name,
                'prompt':      obj.get('instruction', obj.get('question', '')),
                'solution':    obj.get('output', obj.get('response', obj.get('code', ''))),
                'test_code':   '',
                'entry_point': '',
                'difficulty':  'medium',
                'language':   'python',
                'tags':        ['instruction', 'code-generation'],
            })
    return records

# ── CodexEval / BigCodeBench format (same as HumanEval) ─────────────────
def load_humaneval_like(path, source_name):
    """Same schema as HumanEval."""
    return load_humaneval(path)

# ── Train/Val split ──────────────────────────────────────────────────────
def split_train_val(records, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(records)
    n = int(len(records) * val_ratio)
    return records[n:], records[:n]

# ── Build unified dataset ─────────────────────────────────────────────────
def build_dataset():
    all_records = []

    # HumanEval
    he = os.path.join(DATA, 'HumanEval.jsonl')
    if os.path.exists(he):
        print(f'[+] HumanEval: {he}')
        all_records.extend(load_humaneval(he))

    # Synthetic variations
    sv = os.path.join(DATA, 'synthetic_variations.jsonl')
    if os.path.exists(sv):
        print(f'[+] synthetic_variations: {sv}')
        all_records.extend(load_synthetic(sv))

    # APPS
    apps = os.path.join(DATA, 'APPS_train_sample.jsonl')
    if os.path.exists(apps):
        print(f'[+] APPS: {apps}')
        all_records.extend(load_apps(apps))

    # CodexEval
    cx = os.path.join(DATA, 'CodexEval.jsonl')
    if os.path.exists(cx):
        print(f'[+] CodexEval: {cx}')
        all_records.extend(load_humaneval_like(cx, 'codexeval'))

    # CodeAlpaca
    ca = os.path.join(DATA, 'CodeAlpaca.jsonl')
    if os.path.exists(ca):
        print(f'[+] CodeAlpaca: {ca}')
        all_records.extend(load_instruction(ca, 'codealpaca'))

    # Magicoder
    mg = os.path.join(DATA, 'MagicoderEvol.jsonl')
    if os.path.exists(mg):
        print(f'[+] MagicoderEvol: {mg}')
        all_records.extend(load_instruction(mg, 'magicoder'))

    print(f'\nTotal records loaded: {len(all_records)}')

    # Deduplicate by id (keep first)
    seen = set()
    unique = []
    for r in all_records:
        key = r['id']  # rough dedup
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f'After dedup: {len(unique)}')

    # Train/Val split
    train, val = split_train_val(unique, val_ratio=0.1)
    print(f'Train: {len(train)}, Val: {len(val)}')

    # Save
    for records, name in [(train, 'train'), (val, 'val')]:
        out_path = os.path.join(DATA, f'unified_{name}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        sz = os.path.getsize(out_path) / 1024
        print(f'  Saved: {out_path} ({len(records)} rec, {sz:.1f} KB)')

    # Also save full unified
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'  Saved: {OUT} ({len(unique)} rec, {os.path.getsize(OUT)/1024:.1f} KB)')

    # Stats by source
    print('\nBy source:')
    by_src = {}
    for r in unique:
        by_src.setdefault(r['source'], []).append(r)
    for src, recs in sorted(by_src.items(), key=lambda x: -len(x[1])):
        print(f'  {src:<30} {len(recs):>5} records')

    return unique

if __name__ == '__main__':
    build_dataset()
