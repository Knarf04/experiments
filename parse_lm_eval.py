import argparse
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help="Directory to search for JSON files")
parser.add_argument("keyword", type=str, help="Keyword to match filenames")
args = parser.parse_args()

directory = args.directory
keyword = args.keyword.lower()

subtasks = ["lcc", "repobench-p", "lsht", "samsum", "trec", "triviaqa", "2wikimqa", 
            "dureader", "hotpotqa", "musique", "multifieldqa_en", "multifieldqa_zh", 
            "narrativeqa", "qasper", "gov_report", "multi_news", "qmsum", "vcsum",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh"]

matched = sorted(f for f in os.listdir(directory) if f.endswith(".json") and keyword in f.lower())

for fname in matched:
    m = re.search(r"step=(\d+)_", fname)
    step = int(m.group(1)) if m else None
    path = os.path.join(directory, fname)
    with open(path) as f:
        data = json.load(f)
    
    print(f"--- step={step} ---")
    results = data["results"]
    val_list = []
    for subtask in subtasks:
        val = next(v for k, v in results[f"longbench_{subtask}"].items() if k.endswith("score,none"))
        val_list.append(val*100)
    print("\t".join(f"{v:.2f}" for v in val_list))
