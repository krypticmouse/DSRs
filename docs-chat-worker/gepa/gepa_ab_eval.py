"""Paired eval of the GEPA A/B winners: old template + its winner vs new
template + its winner. The end-to-end "does the adapter fix help GEPA" test.

Each arm is the complete system a user would deploy: the arm's ChatAdapter
rendering with the instruction GEPA evolved under that rendering. n=50
questions x 3 repeats, paired per-question deltas (new - old), bootstrap 95%
CI, plus the trap/artifact breakout from scaffold_analysis.py.

    python gepa_ab_eval.py <old_winner.txt> <new_winner.txt>
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run import HERE, load_env
from scaffold_analysis import ARTIFACT, TRAPS, boot_ci
from scaffold_experiment import call_toast, strip_markers

REPEATS = 3
WORKERS = 6
OBJECTIVE_MARK = "In adhering to this structure, your objective is:"


def build_arm(template_path, winner_instruction, old_style):
    raw = Path(template_path).read_text()
    sys_part = raw.split("=== ROLE: System ===")[1].split("=== ROLE: User ===")[0].strip()
    user_tmpl = raw.split("=== ROLE: User ===")[1].strip()
    head = sys_part.split(OBJECTIVE_MARK)[0]
    if old_style:
        indented = "".join(f"\n        {line}" for line in winner_instruction.splitlines())
        system = f"{head}{OBJECTIVE_MARK} {indented}"
    else:
        system = f"{head}{OBJECTIVE_MARK}\n{winner_instruction}"
    return system, user_tmpl


def main():
    load_env()
    import os

    from judge import judge

    old_winner = Path(sys.argv[1]).read_text().strip()
    new_winner = Path(sys.argv[2]).read_text().strip()
    api_key = os.environ["MXBAI_API_KEY"]

    arms = {
        "old": build_arm(HERE / "rendered_chat_baseline.txt", old_winner, old_style=True),
        "new": build_arm(HERE / "rendered_chat.txt", new_winner, old_style=False),
    }

    items = []
    for name in ("evalset.jsonl", "holdout.jsonl"):
        for line in (HERE / name).read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    tasks = [
        (idx, arm, rep, item)
        for idx, item in enumerate(items)
        for arm in arms
        for rep in range(REPEATS)
    ]
    print(f"{len(items)} questions x 2 arms x {REPEATS} reps = {len(tasks)} rollouts")

    def run_one(task):
        idx, arm, rep, item = task
        system, user_tmpl = arms[arm]
        q = item["question"]
        for attempt in range(2):
            try:
                answer = strip_markers(
                    call_toast(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user_tmpl.replace("___QUESTION___", q)}],
                        api_key,
                    )
                )
                score, _, _ = judge(q, answer, item["key_points"], item.get("citations", []))
                return {"q_idx": idx, "arm": arm, "rep": rep, "score": score, "chars": len(answer)}
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    return {"q_idx": idx, "arm": arm, "rep": rep, "score": None, "error": str(e)[:200]}
        return None

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, tasks):
            results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"progress {done}/{len(tasks)}", flush=True)

    out = HERE / "gepa_ab_results.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    ok = [r for r in results if r.get("score") is not None]
    print(f"\nfailed rollouts: {len(results) - len(ok)}")
    per_q = {}
    for r in ok:
        per_q.setdefault(r["q_idx"], {}).setdefault(r["arm"], []).append(r["score"])
    deltas = {}
    for qi, a in sorted(per_q.items()):
        if "old" in a and "new" in a:
            old = sum(a["old"]) / len(a["old"])
            new = sum(a["new"]) / len(a["new"])
            deltas[qi] = (old, new, new - old)

    def group(label, idx):
        sel = [deltas[i][2] for i in sorted(idx) if i in deltas]
        if not sel:
            return
        mean = sum(sel) / len(sel)
        if len(sel) >= 3:
            lo, hi = boot_ci(sel)
            print(f"{label:>14}: delta {mean:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  (n={len(sel)})")
        else:
            print(f"{label:>14}: delta {mean:+.3f}  (n={len(sel)})")

    all_idx = set(deltas)
    print(f"{'old arm mean':>14}: {sum(v[0] for v in deltas.values()) / len(deltas):.3f}")
    print(f"{'new arm mean':>14}: {sum(v[1] for v in deltas.values()) / len(deltas):.3f}")
    group("overall", all_idx)
    group("excl artifact", all_idx - ARTIFACT)
    group("normal", all_idx - ARTIFACT - TRAPS)
    group("traps", all_idx & TRAPS)
    group("artifact", all_idx & ARTIFACT)
    print(f"raw results in {out}")


if __name__ == "__main__":
    main()
