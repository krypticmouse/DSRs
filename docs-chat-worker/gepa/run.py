"""Run GEPA over the docs-chat system prompt.

    python run.py --evalset evalset.jsonl --budget 400

The seed prompt is extracted from ../worker.js so the run always starts from
what is actually deployed. Deploy the result by pasting best_prompt.txt into
SYSTEM in worker.js.
"""

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_env():
    env = HERE.parents[1] / ".env"
    import os

    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def seed_from_worker():
    src = (HERE.parent / "worker.js").read_text()
    m = re.search(r"const SYSTEM = `(.*?)`;", src, re.S)
    if m:
        # undo JS template-literal line continuations
        return m.group(1).replace("\\\n", "")
    m = re.search(r'const SYSTEM = ("(?:[^"\\]|\\.)*");', src)
    if not m:
        raise SystemExit("could not find `const SYSTEM = ...` in worker.js")
    return json.loads(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default=HERE / "evalset.jsonl", type=Path)
    ap.add_argument("--budget", default=400, type=int, help="max metric calls (1 call = 1 toast-1 rollout + 1 judge call)")
    ap.add_argument("--val-frac", default=0.25, type=float)
    ap.add_argument("--workers", default=4, type=int)
    args = ap.parse_args()

    load_env()
    import gepa

    from adapter import DocsChatAdapter

    items = [json.loads(l) for l in args.evalset.read_text().splitlines() if l.strip()]
    random.Random(0).shuffle(items)
    n_val = max(3, int(len(items) * args.val_frac))
    valset, trainset = items[:n_val], items[n_val:]
    print(f"{len(trainset)} train / {len(valset)} val")

    run_dir = HERE / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    result = gepa.optimize(
        seed_candidate={"system_prompt": seed_from_worker()},
        trainset=trainset,
        valset=valset,
        adapter=DocsChatAdapter(max_workers=args.workers),
        reflection_lm="openrouter/openai/gpt-5.6-sol",
        reflection_minibatch_size=3,
        max_metric_calls=args.budget,
        run_dir=str(run_dir),
        track_best_outputs=True,
        display_progress_bar=False,
        seed=0,
    )

    best = result.best_candidate["system_prompt"]
    print("\n=== val scores per candidate ===")
    for i, s in enumerate(result.val_aggregate_scores):
        marker = " <- best" if i == result.best_idx else ""
        print(f"  candidate {i}: {s:.3f}{marker}")
    out = HERE / "best_prompt.txt"
    out.write_text(best)
    print(f"\nbest prompt written to {out}")
    print(f"run artifacts in {run_dir}")


if __name__ == "__main__":
    main()
