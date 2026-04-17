"""Re-convert HumanEval with canonical_solution"""
import sys, os, json
sys.path.insert(0, r'D:\pylib')
import pandas as pd

df = pd.read_parquet(r'D:\IdeaProjects\decentral-ai\data\openai_humaneval\test-00000-of-00001.parquet')
print(f'Columns: {list(df.columns)}')

out_path = r'D:\IdeaProjects\decentral-ai\data\HumanEval.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        obj = {
            'task_id': row['task_id'],
            'prompt': row['prompt'],
            'canonical_solution': row.get('canonical_solution', ''),
            'test': row.get('test', ''),
            'entry_point': row.get('entry_point', '')
        }
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

# Verify
count = 0
with_solutions = 0
with open(out_path, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            count += 1
            if d.get('canonical_solution'):
                with_solutions += 1
print(f'Saved {count} problems, {with_solutions} with canonical solutions')
