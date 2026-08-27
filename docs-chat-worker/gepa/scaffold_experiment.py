"""Does ChatAdapter's typed-contract rendering beat a bare system prompt?

Both arms carry the SAME seed instruction; the only variable is the prompt
scaffolding around it:
  bare     — instruction as system prompt, question as user message
  rendered — DSRs ChatAdapter output (field contract, [[ ## ]] markers,
             instruction at the end), captured verbatim in rendered_chat.txt

n=50 questions x 3 repeats x 2 arms, paired per-question analysis with a
bootstrap 95% CI over questions. Neither arm is optimized, so using the
full evalset + holdout is contamination-free.

    python scaffold_experiment.py            # ~300 rollouts, ~35 min
"""

import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run import HERE, load_env

REPEATS = 3
WORKERS = 6

BARE_SYSTEM = """You are the DSRs docs assistant. DSRs (dspy-rs) is a Rust \
framework for building and optimizing LM pipelines. Answer only from the \
attached stores: dsrs-docs (published documentation) and dsrs-code (the \
Rust sources under crates/). Search before answering; use grep for exact \
symbol or signature lookups. When an answer touches implementation, quote \
the real code with its file path. If the stores don't cover the question, \
say so instead of guessing. Answer in concise markdown."""


def load_rendered():
    raw = (HERE / "rendered_chat.txt").read_text()
    sys_part = raw.split("=== ROLE: System ===")[1].split("=== ROLE: User ===")[0].strip()
    user_tmpl = raw.split("=== ROLE: User ===")[1].strip()
    return sys_part, user_tmpl


def call_toast(messages, api_key):
    req = urllib.request.Request(
        "https://api.mixedbread.com/v1/chat/completions",
        data=json.dumps(
            {
                "model": "toast-1",
                "stream": False,
                "max_tokens": 2048,
                "messages": messages,
                "tools": [
                    {"type": "store_search", "store_identifiers": ["dsrs-docs", "dsrs-code"]},
                    {"type": "store_grep", "store_identifiers": ["dsrs-docs", "dsrs-code"]},
                ],
            }
        ).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    return resp["choices"][0]["message"].get("content") or ""


def strip_markers(text):
    s = text.split("[[ ## answer ## ]]")[-1]
    return s.split("[[ ## completed ## ]]")[0].strip()


def main():
    load_env()
    import os

    from judge import judge

    api_key = os.environ["MXBAI_API_KEY"]
    sys_part, user_tmpl = load_rendered()

    items = []
    for name in ("evalset.jsonl", "holdout.jsonl"):
        for line in (HERE / name).read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    print(f"{len(items)} questions x {REPEATS} repeats x 2 arms = {len(items) * REPEATS * 2} rollouts")

    def run_one(task):
        idx, arm, rep, item = task
        q = item["question"]
        for attempt in range(2):
            try:
                if arm == "bare":
                    answer = call_toast(
                        [{"role": "system", "content": BARE_SYSTEM},
                         {"role": "user", "content": q}],
                        api_key,
                    )
                else:
                    answer = strip_markers(
                        call_toast(
                            [{"role": "system", "content": sys_part},
                             {"role": "user", "content": user_tmpl.replace("___QUESTION___", q)}],
                            api_key,
                        )
                    )
                score, _, _ = judge(q, answer, item["key_points"], item.get("citations", []))
                return {"q_idx": idx, "arm": arm, "rep": rep, "score": score, "chars": len(answer)}
            except Exception as e:
                if attempt == 1:
                    return {"q_idx": idx, "arm": arm, "rep": rep, "score": None, "error": str(e)[:200]}
        return None

    tasks = [
        (idx, arm, rep, item)
        for idx, item in enumerate(items)
        for arm in ("bare", "rendered")
        for rep in range(REPEATS)
    ]
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, tasks):
            results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"progress {done}/{len(tasks)}", flush=True)

    out = HERE / "scaffold_results.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    # paired analysis: per-question repeat-averages, delta = rendered - bare
    ok = [r for r in results if r.get("score") is not None]
    failed = len(results) - len(ok)
    per_q = {}
    for r in ok:
        per_q.setdefault(r["q_idx"], {}).setdefault(r["arm"], []).append(r["score"])
    deltas, bare_means, rend_means = [], [], []
    for qi, arms in sorted(per_q.items()):
        if "bare" in arms and "rendered" in arms:
            b = sum(arms["bare"]) / len(arms["bare"])
            re_ = sum(arms["rendered"]) / len(arms["rendered"])
            bare_means.append(b)
            rend_means.append(re_)
            deltas.append(re_ - b)

    n = len(deltas)
    mean_delta = sum(deltas) / n
    rng = random.Random(0)
    boots = []
    for _ in range(10_000):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo, hi = boots[249], boots[9749]
    wins = sum(1 for d in deltas if d > 0.02)
    losses = sum(1 for d in deltas if d < -0.02)

    print(f"\nfailed rollouts: {failed}")
    print(f"n questions paired: {n}")
    print(f"bare mean:     {sum(bare_means) / n:.3f}")
    print(f"rendered mean: {sum(rend_means) / n:.3f}")
    print(f"paired delta (rendered - bare): {mean_delta:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]")
    print(f"question wins/losses/ties (|d|>0.02): {wins}/{losses}/{n - wins - losses}")
    verdict = (
        "RENDERED WINS" if lo > 0 else "BARE WINS" if hi < 0 else "NO SIGNIFICANT DIFFERENCE"
    )
    print(f"verdict: {verdict}")
    print(f"raw results in {out}")


if __name__ == "__main__":
    main()
