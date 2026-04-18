import os
import json

jsonl_path = r"D:\IdeaProjects\decentral-ai\results\test_5.jsonl"
if os.path.exists(jsonl_path):
    size = os.path.getsize(jsonl_path)
    print(f"File size: {size:,} bytes")
    if size > 0:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Lines: {len(lines)}")
        if lines:
            data = json.loads(lines[-1])
            print(f"Last entry: task_id={data.get('task_id')}, completion_tokens={data.get('completion_tokens')}")
else:
    print("File not found")
