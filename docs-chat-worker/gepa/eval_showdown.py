"""Paired holdout showdown across N prompt arms.

Each arm is a complete system prompt deployed worker-style (system prompt +
raw user question, same store tools as production — adapter.call_student).
Arms are name=path pairs; the special value SEED extracts the currently
deployed prompt from ../worker.js the way run.py does.

    python eval_showdown.py \
        --arm seed=SEED \
        --arm pygepa=best_prompt.txt \
        --arm dsrs=dsrs_prompt.txt \
        --ref seed [--holdout holdout40.jsonl] [--reps 2]

Per non-ref arm, against --ref on the shared questions:
  - paired per-question delta with a bootstrap 95% CI (10k resamples, seed 0)
  - exact two-sided McNemar on success = per-question repeat-mean >= --threshold
    (default 0.7 on the judge's 0-1 scale; the August campaign aggregated raw
    means, so the threshold test is reported alongside, not instead)
  - answer length (mean chars)
  - leak audit: verbatim >= 8-word n-grams shared between the arm's prompt
    and any train/val question or key point (--train/--val)

Raw rows go to --out as jsonl: {"q_idx", "arm", "rep", "score", "chars"}.
"""

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run import HERE, load_env, seed_from_worker

NGRAM = 8


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def strip_markers(text):
    s = text.split("[[ ## answer ## ]]")[-1]
    return s.split("[[ ## completed ## ]]")[0].strip()


def boot_ci(values, n_boot=10_000, seed=0):
    import random

    n = len(values)
    rng = random.Random(seed)
    boots = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return boots[int(n_boot * 0.025)], boots[int(n_boot * 0.975) - 1]


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p on discordant counts (b: ref-only success,
    c: arm-only success). Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def word_ngrams(text, n=NGRAM):
    words = re.sub(r"\s+", " ", text.lower()).split(" ")
    words = [w.strip("`*_\"'.,;:!?()[]{}") for w in words]
    words = [w for w in words if w]
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def leak_audit(prompt_text, sources):
    """sources: list of (label, string). Returns (leaked source strings,
    example n-grams)."""
    prompt_grams = word_ngrams(prompt_text)
    leaks, examples = [], []
    for label, s in sources:
        shared = word_ngrams(s) & prompt_grams
        if shared:
            leaks.append(label)
            examples.append((label, " ".join(next(iter(shared)))))
    return leaks, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PROMPT_FILE",
        help="arm name and prompt file; use NAME=SEED for the worker.js prompt",
    )
    ap.add_argument("--ref", required=True, help="reference arm name for paired stats")
    ap.add_argument("--holdout", default=HERE / "holdout40.jsonl", type=Path)
    ap.add_argument("--train", default=HERE / "train50.jsonl", type=Path)
    ap.add_argument("--val", default=HERE / "val30.jsonl", type=Path)
    ap.add_argument("--reps", default=2, type=int)
    ap.add_argument("--workers", default=6, type=int)
    ap.add_argument("--threshold", default=0.7, type=float)
    ap.add_argument("--out", default=HERE / "showdown_results.jsonl", type=Path)
    args = ap.parse_args()

    load_env()
    from adapter import call_student
    from judge import judge

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"bad --arm {spec!r}, expected NAME=PROMPT_FILE")
        arms[name] = seed_from_worker() if path == "SEED" else Path(path).read_text().strip()
    if args.ref not in arms:
        raise SystemExit(f"--ref {args.ref!r} is not an arm ({', '.join(arms)})")

    items = load_rows(args.holdout)
    tasks = [
        (idx, arm, rep, item)
        for idx, item in enumerate(items)
        for arm in arms
        for rep in range(args.reps)
    ]
    print(f"{len(items)} questions x {len(arms)} arms x {args.reps} reps = {len(tasks)} rollouts")

    def run_one(task):
        idx, arm, rep, item = task
        q = item["question"]
        try:
            answer, _ = call_student(arms[arm], q)
            answer = strip_markers(answer)
            score, _, _ = judge(q, answer, item["key_points"], item.get("citations", []))
            return {"q_idx": idx, "arm": arm, "rep": rep, "score": score, "chars": len(answer)}
        except Exception as e:  # noqa: BLE001
            return {"q_idx": idx, "arm": arm, "rep": rep, "score": None, "error": str(e)[:200]}

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_one, tasks):
            results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"progress {done}/{len(tasks)}", flush=True)

    args.out.write_text("".join(json.dumps(r) + "\n" for r in results))
    ok = [r for r in results if r.get("score") is not None]
    print(f"failed rollouts: {len(results) - len(ok)}  (raw rows in {args.out})")

    # per-question repeat-means per arm
    per_q = {}
    for r in ok:
        per_q.setdefault(r["q_idx"], {}).setdefault(r["arm"], []).append(r)
    means = {
        qi: {
            arm: sum(x["score"] for x in rows) / len(rows)
            for arm, rows in by_arm.items()
        }
        for qi, by_arm in per_q.items()
    }

    # leak audit sources: every train/val question and key point
    sources = []
    for path in (args.train, args.val):
        for i, row in enumerate(load_rows(path)):
            sources.append((f"{path.name}:{i}:question", row["question"]))
            for j, kp in enumerate(row["key_points"]):
                sources.append((f"{path.name}:{i}:kp{j}", kp))

    print(f"\n{'arm':>10}  {'mean':>6}  {'>= {:.2f}'.format(args.threshold):>8}  "
          f"{'ans chars':>9}  {'prompt chars':>12}  {'leaks':>5}")
    for arm in arms:
        arm_means = [m[arm] for m in means.values() if arm in m]
        chars = [r["chars"] for r in ok if r["arm"] == arm]
        succ = sum(1 for m in arm_means if m >= args.threshold)
        leaks, _ = leak_audit(arms[arm], sources)
        print(f"{arm:>10}  {sum(arm_means)/len(arm_means):>6.3f}  "
              f"{succ:>4}/{len(arm_means):<3}  {sum(chars)/len(chars):>9.0f}  "
              f"{len(arms[arm]):>12}  {len(leaks):>5}")

    ref = args.ref
    for arm in arms:
        if arm == ref:
            continue
        shared = sorted(qi for qi, m in means.items() if arm in m and ref in m)
        deltas = [means[qi][arm] - means[qi][ref] for qi in shared]
        mean_d = sum(deltas) / len(deltas)
        lo, hi = boot_ci(deltas)
        b = sum(1 for qi in shared
                if means[qi][ref] >= args.threshold > means[qi][arm])
        c = sum(1 for qi in shared
                if means[qi][arm] >= args.threshold > means[qi][ref])
        p = mcnemar_exact(b, c)
        verdict = f"{arm} WINS" if lo > 0 else f"{ref} WINS" if hi < 0 else "NO SIG DIFF"
        print(f"\n== {arm} vs {ref} (n={len(shared)} questions) ==")
        print(f"paired delta {mean_d:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  -> {verdict}")
        print(f"McNemar on score >= {args.threshold}: {arm}-only wins {c}, "
              f"{ref}-only wins {b}, exact p = {p:.4f}")

    print("\n== leak audit (verbatim >= 8-word n-grams vs train/val) ==")
    for arm in arms:
        leaks, examples = leak_audit(arms[arm], sources)
        if not leaks:
            print(f"{arm}: clean")
            continue
        print(f"{arm}: {len(leaks)} leaked train/val strings")
        for label, gram in examples[:10]:
            print(f'  {label}: "{gram} ..."')


if __name__ == "__main__":
    main()
