"""Breakout analysis for scaffold_experiment.py results.

Reports the paired rendered-vs-bare delta overall and per question category:
  normal    — ordinary docs questions
  traps     — the 3 "not in the docs" refusal questions (42, 43, 49)
  artifact  — q12 asks the model to *describe* the marker protocol, so its
              correct answer contains the markers post-processing strips;
              it is excluded from the headline number

Acceptance test for the refusal-safe response instruction (adapter roadmap
§5.1): traps delta >= 0.

    python scaffold_analysis.py [results.jsonl] [baseline.jsonl]

With a second path, also prints the same breakdown for the baseline file so
runs are comparable side by side.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
TRAPS = {42, 43, 49}
ARTIFACT = {12}


def load(path):
    per_q = {}
    failed = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("score") is None:
            failed += 1
            continue
        per_q.setdefault(r["q_idx"], {}).setdefault(r["arm"], []).append(r["score"])
    deltas = {}
    for qi, arms in sorted(per_q.items()):
        if "bare" in arms and "rendered" in arms:
            bare = sum(arms["bare"]) / len(arms["bare"])
            rend = sum(arms["rendered"]) / len(arms["rendered"])
            deltas[qi] = (bare, rend, rend - bare)
    return deltas, failed


def boot_ci(values, n_boot=10_000, seed=0):
    n = len(values)
    rng = random.Random(seed)
    boots = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return boots[int(n_boot * 0.025)], boots[int(n_boot * 0.975) - 1]


def report(name, deltas):
    def stats(indices):
        sel = [deltas[i][2] for i in indices if i in deltas]
        if not sel:
            return None
        mean = sum(sel) / len(sel)
        return mean, len(sel)

    print(f"\n== {name} ==")
    all_idx = set(deltas)
    groups = [
        ("overall", all_idx),
        ("excl artifact", all_idx - ARTIFACT),
        ("normal", all_idx - ARTIFACT - TRAPS),
        ("traps", all_idx & TRAPS),
        ("artifact", all_idx & ARTIFACT),
    ]
    for label, idx in groups:
        s = stats(idx)
        if s is None:
            continue
        mean, n = s
        sel = [deltas[i][2] for i in sorted(idx) if i in deltas]
        if n >= 3:
            lo, hi = boot_ci(sel)
            print(f"{label:>14}: delta {mean:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  (n={n})")
        else:
            per = ", ".join(f"q{i}: {deltas[i][2]:+.3f}" for i in sorted(idx) if i in deltas)
            print(f"{label:>14}: delta {mean:+.3f}  (n={n}: {per})")
    bare = sum(v[0] for v in deltas.values()) / len(deltas)
    rend = sum(v[1] for v in deltas.values()) / len(deltas)
    print(f"{'means':>14}: bare {bare:.3f}  rendered {rend:.3f}")
    traps_mean = stats(all_idx & TRAPS)
    if traps_mean is not None:
        verdict = "PASS" if traps_mean[0] >= 0 else "FAIL"
        print(f"{'acceptance':>14}: traps delta >= 0 -> {verdict}")


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else HERE / "scaffold_results.jsonl"
    deltas, failed = load(results)
    if failed:
        print(f"({failed} failed rollouts excluded)")
    report(str(results), deltas)
    if len(sys.argv) > 2:
        base_deltas, base_failed = load(sys.argv[2])
        if base_failed:
            print(f"({base_failed} failed rollouts excluded)")
        report(sys.argv[2], base_deltas)


if __name__ == "__main__":
    main()
