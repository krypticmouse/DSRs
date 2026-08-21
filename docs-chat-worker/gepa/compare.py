"""Compare two system prompts on a holdout set the optimizer never saw.

    python compare.py --a seed --b best_prompt.txt --holdout holdout.jsonl

--a/--b accept 'seed' (extract from ../worker.js) or a path to a prompt file.
Each question runs once per prompt; the judge scores both. Prints per-question
scores and the mean delta.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run import HERE, load_env, seed_from_worker


def read_prompt(spec):
    return seed_from_worker() if spec == "seed" else Path(spec).read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="seed")
    ap.add_argument("--b", default=HERE / "best_prompt.txt")
    ap.add_argument("--c", default=None, help="optional third arm")
    ap.add_argument("--holdout", default=HERE / "holdout.jsonl", type=Path)
    ap.add_argument("--workers", default=4, type=int)
    args = ap.parse_args()

    load_env()
    from adapter import call_student
    from judge import judge

    prompts = {"A": read_prompt(str(args.a)), "B": read_prompt(str(args.b))}
    if args.c:
        prompts["C"] = read_prompt(str(args.c))
    items = [json.loads(l) for l in args.holdout.read_text().splitlines() if l.strip()]

    def run_one(task):
        label, item = task
        try:
            answer, _ = call_student(prompts[label], item["question"])
            score, feedback, _ = judge(
                item["question"], answer, item["key_points"], item.get("citations", [])
            )
        except Exception as e:
            answer, score, feedback = "", 0.0, f"ERROR: {e}"
        return label, item["question"], score, len(answer), feedback

    tasks = [(label, item) for item in items for label in sorted(prompts)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, tasks))

    by_q = {}
    for label, q, score, chars, feedback in results:
        by_q.setdefault(q, {})[label] = (score, chars, feedback)

    labels = sorted(prompts)
    header = " ".join(f"{l:>6}" for l in labels)
    print(f"\n{'Q':<64} {header}")
    sums = {l: 0.0 for l in labels}
    for q, r in by_q.items():
        row = " ".join(f"{r.get(l, (0, 0, ''))[0]:>6.2f}" for l in labels)
        for l in labels:
            sums[l] += r.get(l, (0, 0, ""))[0]
        print(f"{q[:62]:<64} {row}")
    means = " | ".join(f"{l}: {sums[l] / len(by_q):.3f}" for l in labels)
    print(f"\nmeans -> {means}")

    detail = HERE / "compare_detail.json"
    detail.write_text(json.dumps(by_q, indent=2))
    print(f"per-question feedback in {detail}")


if __name__ == "__main__":
    main()
