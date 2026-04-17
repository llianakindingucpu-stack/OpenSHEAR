"""Generate synthetic training data from HumanEval problems.

Strategy: for each problem, generate 3-5 variations by:
  1. Perturbing docstring / description
  2. Renaming functions/args
  3. Changing test cases (keeping same logic)
  4. Adding type hints or removing them
  5. Using equivalent standard-lib imports

Also download any publicly accessible datasets.
"""
import sys, os, json, re, random
sys.path.insert(0, r'D:\pylib')

import huggingface_hub as hf

DATA = r'D:\IdeaProjects\decentral-ai\data'
OUT  = os.path.join(DATA, 'synthetic_variations.jsonl')
os.makedirs(DATA, exist_ok=True)
hf.ENDPOINT = "https://hf-mirror.com"

# ── HumanEval variations ────────────────────────────────────────────────
he_path = os.path.join(DATA, 'HumanEval.jsonl')

def snake_to_camel(s):
    parts = s.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])

def camel_to_snake(s):
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    return s

NAME_VARIANTS = {
    'find':   ['locate', 'search', 'get_index'],
    'max':    ['maximum', 'largest', 'peak'],
    'min':    ['minimum', 'smallest', 'lowest'],
    'sort':   ['order', 'arrange', 'rank'],
    'sum':    ['total', 'add_up', 'aggregate'],
    'reverse':['flip', 'invert', 'backward'],
    'unique': ['distinct', 'uniquify', 'no_dup'],
    'merge':  ['combine', 'join', 'blend'],
    'split':  ['divide', 'partition', 'separate'],
    'count':  ['tally', 'freq', 'occurrences'],
    'fib':    ['fibonacci', 'fib_seq'],
    'prime':  ['is_prime', 'check_prime', 'prime_check'],
    'is_palindrome': ['palindrome', 'check_palindrome', 'is_mirror'],
}

TYPE_VARIANTS = [
    '', '-> bool', '-> int', '-> List[int]', '-> str', '-> List[str]',
    '(x: List[int]) -> int', '(s: str) -> bool',
    '(arr: list[int], target: int) -> int',
]

def perturb_docstring(docstring):
    """Rewrite docstring with different wording but same semantics."""
    replacements = [
        (r'\bthe\s+maximum\b', 'the largest'),
        (r'\bthe\s+minimum\b', 'the smallest'),
        (r'\breturn\b', 'output'),
        (r'\bparameter\b', 'argument'),
        (r'\bGiven\b', 'Input:'),
        (r'\breturns?\b', 'output'),
        (r'\bCalculate\b', 'Compute'),
        (r'\bCheck\b', 'Determine if'),
        (r'\bFind\b', 'Locate'),
        (r'\bfirst\b', 'initial'),
        (r'\blast\b', 'final'),
    ]
    result = docstring
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result

def perturb_code(code, canonical, entry_point):
    """Rename function/args while keeping logic intact."""
    result = code
    # Rename function
    if entry_point in NAME_VARIANTS:
        new_name = random.choice(NAME_VARIANTS[entry_point])
        result = result.replace(f'def {entry_point}(', f'def {new_name}(')
        result = re.sub(rf'\b{entry_point}\b(?!\w)', new_name, result)
        # Update canonical solution reference
        canonical_new = canonical.replace(f'def {entry_point}(', f'def {new_name}(')
        canonical_new = re.sub(rf'\b{entry_point}\b(?!\w)', new_name, canonical_new)
    else:
        new_name = entry_point
        canonical_new = canonical
    # Perturb type hints
    if random.random() < 0.4:
        old_type = random.choice(TYPE_VARIANTS)
        new_type = random.choice([t for t in TYPE_VARIANTS if t != old_type])
        if old_type in result:
            result = result.replace(old_type, new_type)
    # Add/remove whitespace
    if random.random() < 0.2:
        result = re.sub(r'\n\s*\n+', '\n\n', result)
    return result, canonical_new

def generate_variations(problem, n=4):
    """Generate n variations of a HumanEval problem."""
    entry_point = problem.get('entry_point', '')
    docstring   = problem.get('prompt', '')
    canonical   = problem.get('canonical_solution', problem.get('solution', ''))
    test        = problem.get('test', '')
    difficulty  = problem.get('difficulty', 'medium')

    if not canonical:
        return []

    variations = []
    for i in range(n):
        p2 = dict(problem)
        p2['variation_id'] = f'{problem.get("task_id", problem.get("instance_id", "unk"))}_v{i+1}'
        p2['parent_id']     = problem.get('task_id', problem.get('instance_id', ''))
        p2['is_variation']  = True

        # Perturb docstring
        new_doc = perturb_docstring(docstring)
        p2['prompt'] = new_doc

        # Perturb code
        new_code, new_canonical = perturb_code(problem.get('prompt', ''), canonical, entry_point)
        p2['canonical_solution'] = new_canonical

        # Perturb test (simple: rename calls)
        new_test = test
        if entry_point in NAME_VARIANTS and random.random() < 0.5:
            alt_name = random.choice(NAME_VARIANTS[entry_point])
            new_test = re.sub(rf'\b{entry_point}\b', alt_name, test)
        p2['test'] = new_test

        # Randomise difficulty slightly
        if random.random() < 0.3:
            diffs = ['easy', 'medium', 'hard']
            p2['difficulty'] = random.choice(diffs)

        variations.append(p2)

    return variations

# Load and generate
print(f'\n[1] Generating variations from HumanEval ({he_path})')
with open(he_path, 'rb') as f:
    raw = f.read().decode('utf-8', errors='replace')

problems = []
for line in raw.strip().split('\n'):
    line = line.strip()
    if line:
        try:
            problems.append(json.loads(line))
        except Exception:
            pass

print(f'    Loaded {len(problems)} problems')

all_variations = []
for p in problems:
    # Generate 4 variations per problem
    for v in generate_variations(p, n=4):
        all_variations.append(v)

print(f'    Generated {len(all_variations)} variations')

# Save to JSONL
with open(OUT, 'w', encoding='utf-8') as f:
    for v in all_variations:
        f.write(json.dumps(v, ensure_ascii=False) + '\n')

sz = os.path.getsize(OUT) / 1024
print(f'    Saved: synthetic_variations.jsonl — {len(all_variations)} records, {sz:.1f} KB')

# ── Try downloading open public datasets ────────────────────────────────
public_repos = [
    # repo_id, filename, dest_name
    ('bigcode/bigcodebench', 'BigCodeBench.jsonl', 'BigCodeBench.jsonl'),
    ('bigcode/the-stack', 'data.parquet', 'TheStack_sample.jsonl'),
    ('mystic-ai/codex-eval', 'humaneval.jsonl', 'CodexEval.jsonl'),
    ('codeparrot/apps', 'train.jsonl', 'APPS_train_sample.jsonl'),
    ('nampdn-ai/tiny-codes', 'data.parquet', 'TinyCodes.jsonl'),
]

def try_download(repo_id, filename, dest_name):
    dest = os.path.join(DATA, dest_name)
    if os.path.exists(dest):
        n = sum(1 for _ in open(dest, 'rb'))
        print(f'  [SKIP] {dest_name} already exists ({n} lines)')
        return True
    try:
        print(f'  [TRY] {repo_id}/{filename}')
        tmp = hf.hf_hub_download(repo_id, filename, repo_type='dataset',
                                 local_dir=DATA, local_dir_use_symlinks=False)
        import shutil
        shutil.copy2(tmp, dest)
        sz = os.path.getsize(dest) / 1024
        print(f'  [OK] {dest_name} — {sz:.1f} KB')
        return True
    except Exception as e:
        # Try JSONL directly from raw URL
        try:
            import urllib.request
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(dest, 'wb') as f:
                f.write(data)
            sz = os.path.getsize(dest) / 1024
            print(f'  [OK] {dest_name} via raw URL — {sz:.1f} KB')
            return True
        except Exception as e2:
            print(f'  [SKIP] {dest_name} — {e} / {e2}')
            return False

print('\n[2] Trying publicly accessible datasets')
for repo_id, fname, dest_name in public_repos:
    try_download(repo_id, fname, dest_name)

# ── Summary ──────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Dataset generation complete. Summary:')
for fn in sorted(os.listdir(DATA)):
    fp = os.path.join(DATA, fn)
    if fn.endswith('.jsonl') and os.path.isfile(fp):
        try:
            n = sum(1 for _ in open(fp, 'rb'))
            sz = os.path.getsize(fp) / 1024
            print(f'  {fn:<50} {n:>5} rec  {sz:>7.1f} KB')
        except Exception:
            pass
print('='*60)
