from collections import Counter
import json

file_path = "evalchemy/MOD_DING_TRAINING_0.4/MATH500/__code__SKIPGPT__models__Llama31-8B-instruct-fullsft-openthoughts__/results_2025-10-28T09-49-31.858085.json"  # JSONL 文件路径
counter = Counter()

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        level = data.get("level", None)
        if level is not None:
            counter[level] += 1

print("各 level 题目数量：")
for lvl, count in sorted(counter.items()):
    print(f"Level {lvl}: {count} 道题")
