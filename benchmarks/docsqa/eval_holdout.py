"""Paired multi-arm eval on a frozen DocsQA split, one fresh session.

  python3 eval_holdout.py <split-name> <passes> <out.jsonl> seed=seed \
      [<label>=<winner.json> ...]

Every arm runs every question in every pass, through one serve process
(byte-identical student+judge). Winner JSONs are {"answerer": text}. Any
label may take the value `seed` (or `SEED`): `seed=seed seed2=seed` runs the
seed twice, interleaved like every other arm, and the verdict_stats block
then prints `seed2 − seed` first as the identical-prompt floor and flags
every other arm clears-floor / within-floor (2026-09-02: two byte-identical
bcplus prompts scored .383 vs .475 at devholdout 60×2).
Summary: per-arm mean judge score + paired bootstrap 95% CI vs the seed arm
on per-question means pooled across passes; also mean answer length proxy.

Transport discipline (2026-08-30 audit: error-as-0 rows flipped two
verdicts — 212/600 zeros in the VOID qwen final, and the "deployed rot"
reading):
  - client failures are retried once with backoff before being recorded;
  - rows that still fail are tagged ``"transport": true`` and EXCLUDED from
    the primary means/CIs (an all-rows mean is printed alongside for
    continuity with older logs);
  - on resume, transport rows are re-queued instead of trusted;
  - the full answer text is persisted per row, so judge audits and leak
    forensics never re-buy student rollouts.
"""

import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import DATA, ServeClient, load_rows


def main():
    split, passes, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    arms = {}
    for spec in sys.argv[4:]:
        label, _, path = spec.partition("=")
        if path in ("", "seed", "SEED"):
            arms[label] = None
        else:
            # Winner JSONs carry every slot ({"answerer": ...} for 36-docsqa,
            # {"searcher": ..., "answerer": ...} for 37-docsqa-qwen).
            arms[label] = json.load(open(path))
    rows = load_rows(DATA / f"{split}.jsonl")
    print(f"{split}: {len(rows)} rows x {passes} passes x {len(arms)} arms")

    # Resume: clean rows already in out_path are trusted and skipped;
    # transport rows (tagged, or legacy "ERROR ..." feedback) are re-queued.
    import os

    results = defaultdict(lambda: defaultdict(list))
    done = set()
    bad = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["arm"], r["idx"], r["pass"])
            transport = bool(r.get("transport")) or str(r.get("feedback", "")).startswith(
                "ERROR"
            )
            if transport:
                bad.add(key)
                continue
            done.add(key)
            results[r["arm"]][r["idx"]].append((r["score"], False))
        bad -= done
        if done or bad:
            print(f"resuming: {len(done)} clean rows on disk, {len(bad)} transport rows re-queued")

    client = ServeClient()
    out = open(out_path, "a")
    jobs = [
        (label, i, p)
        for p in range(passes)
        for i in range(len(rows))
        for label in arms
        if (label, i, p) not in done
    ]

    def run(job):
        label, i, p = job
        last_err = None
        for attempt in range(2):
            try:
                r = client.rollout(split, i, arms[label], timeout=300)
                return label, i, p, r, False
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(5 * (attempt + 1))
        return (
            label,
            i,
            p,
            {"score": 0.0, "feedback": f"ERROR {last_err}", "answer": ""},
            True,
        )

    from concurrent.futures import as_completed

    transport_counts = defaultdict(int)
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(run, job) for job in jobs]
            for future in as_completed(futures):
                label, i, p, r, transport = future.result()
                results[label][i].append((r["score"], transport))
                if transport:
                    transport_counts[label] += 1
                out.write(
                    json.dumps(
                        {
                            "arm": label,
                            "idx": i,
                            "pass": p,
                            "question": rows[i]["question"],
                            "score": r["score"],
                            "answer": r.get("answer", ""),
                            "answer_chars": len(r.get("answer", "")),
                            "feedback": r["feedback"][:400],
                            "transport": transport,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
    finally:
        client.close()
        out.close()

    idxs = list(range(len(rows)))

    def per_question(label, include_transport):
        means = []
        for i in idxs:
            cells = results[label][i]
            kept = [s for s, t in cells if include_transport or not t]
            means.append(sum(kept) / len(kept) if kept else None)
        return means

    rng = random.Random(0)
    print(f"\n== {split} x{passes} (primary = transport rows excluded) ==")
    clean = {label: per_question(label, include_transport=False) for label in arms}
    everything = {label: per_question(label, include_transport=True) for label in arms}
    for label in arms:
        clean_scores = [m for m in clean[label] if m is not None]
        all_scores = [m for m in everything[label] if m is not None]
        line = (
            f"{label:14} mean={sum(clean_scores)/max(1,len(clean_scores)):.4f}"
            f" (all-rows {sum(all_scores)/max(1,len(all_scores)):.4f},"
            f" transport {transport_counts[label]})"
        )
        if label != "seed" and "seed" in clean:
            paired = [
                (a, b)
                for a, b in zip(clean["seed"], clean[label])
                if a is not None and b is not None
            ]
            if paired:
                deltas = [b - a for a, b in paired]
                boots = sorted(
                    sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                    for _ in range(10000)
                )
                lo, hi = boots[249], boots[9749]
                sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
                line += f"  Δ={sum(deltas)/len(deltas):+.4f} [{lo:+.4f},{hi:+.4f}] {sig}"
                if len(paired) < len(idxs):
                    line += f" (paired on {len(paired)}/{len(idxs)} q)"
        print(line)

    # Optional Tier-3 block (L3 §7 item 4): sign-flip permutation + pre-registered
    # CUPED on the seed score, printed AFTER the legacy verdict above. With a
    # seed2 arm the module prints `seed2 − seed` first as the identical-prompt
    # floor and flags every other arm clears-floor / within-floor. Reads the
    # rows just written; a missing/broken module never breaks the eval.
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from verdict_stats import print_verdict_table

        print_verdict_table(out_path, seed_arm="seed")
    except Exception as e:  # noqa: BLE001
        print(f"(verdict_stats block skipped: {e})")


if __name__ == "__main__":
    main()
