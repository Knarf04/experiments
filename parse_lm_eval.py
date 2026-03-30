import argparse
import json
import os
import re

LONGBENCH_SUBTASKS = [
    "lcc", "repobench-p", "lsht", "samsum", "trec", "triviaqa", "2wikimqa",
    "dureader", "hotpotqa", "musique", "multifieldqa_en", "multifieldqa_zh",
    "narrativeqa", "qasper", "gov_report", "multi_news", "qmsum", "vcsum",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
]

# (result_key, metric_key)
LEADERBOARD_SUBTASKS = [
    ("leaderboard_mmlu_pro", "acc,none"),
    ("leaderboard_bbh", "acc_norm,none"),
    ("leaderboard_gpqa", "acc_norm,none"),
    ("leaderboard_math_hard", "exact_match,none"),
    ("leaderboard_ifeval", "prompt_level_strict_acc,none"),
    ("leaderboard_musr", "acc_norm,none"),
]


def parse_longbench(data, step):
    results = data["results"]
    print(f"--- step={step} ---")
    header = "\t".join(LONGBENCH_SUBTASKS)
    val_list = []
    for subtask in LONGBENCH_SUBTASKS:
        val = next(v for k, v in results[f"longbench_{subtask}"].items() if k.endswith("score,none"))
        val_list.append(val * 100)
    print(header)
    print("\t".join(f"{v:.2f}" for v in val_list))


def parse_leaderboard(data, step):
    results = data["results"]
    print(f"--- step={step} ---")
    names = [name for name, _ in LEADERBOARD_SUBTASKS]
    header = "\t".join(names)
    val_list = []
    for name, metric_key in LEADERBOARD_SUBTASKS:
        val = results[name][metric_key]
        val_list.append(val * 100)
    print(header)
    print("\t".join(f"{v:.2f}" for v in val_list))


TASK_MAP = {
    "longbench": parse_longbench,
    "leaderboard": parse_leaderboard,
}

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help="Directory to search for JSON files")
parser.add_argument("keyword", type=str, help="Keyword to match filenames")
parser.add_argument("--task", type=str, default="longbench", choices=TASK_MAP.keys(),
                    help="Which task format to parse (default: longbench)")
args = parser.parse_args()

directory = args.directory
keyword = args.keyword.lower()
parse_fn = TASK_MAP[args.task]

matched = sorted(f for f in os.listdir(directory) if f.endswith(".json") and keyword in f.lower())

for fname in matched:
    m = re.search(r"step=(\d+)_", fname)
    step = int(m.group(1)) if m else None
    path = os.path.join(directory, fname)
    with open(path) as f:
        data = json.load(f)
    parse_fn(data, step)
