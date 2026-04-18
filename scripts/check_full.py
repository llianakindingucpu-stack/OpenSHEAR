import os
import json

jsonl_path = r"D:\IdeaProjects\decentral-ai\results\test_5.jsonl"
if os.path.exists(jsonl_path):
    size = os.path.getsize(jsonl_path)
    print(f"File size: {size:,} bytes")
    if size > 0:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Total entries: {len(lines)}")
        for i, line in enumerate(lines):
            data = json.loads(line)
            print(f"  [{i}] task_id={data.get('task_id')}, completion_tokens={data.get('completion_tokens')}, speed={data.get('speed_tok_per_s', 0):.1f}")
else:
    print("File not found")
