import os
import json

results_dir = r"D:\IdeaProjects\decentral-ai\results"
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    print(f"Files in {results_dir}:")
    for f in files:
        path = os.path.join(results_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size:,} bytes")
else:
    print(f"Directory not found: {results_dir}")

# Check specific file
jsonl_path = os.path.join(results_dir, "humaneval_rwkv169m_rust.jsonl")
if os.path.exists(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"\nTotal lines: {len(lines)}")
    print("\nFirst 3 lines (truncated):")
    for i, line in enumerate(lines[:3]):
        data = json.loads(line)
        print(f"  [{i+1}] task_id={data.get('task_id')}, tokens={data.get('completion_tokens')}")
    print("\nLast 3 lines (truncated):")
    for i, line in enumerate(lines[-3:]):
        data = json.loads(line)
        print(f"  [{len(lines)-2+i}] task_id={data.get('task_id')}, tokens={data.get('completion_tokens')}")
