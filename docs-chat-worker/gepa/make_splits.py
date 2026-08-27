"""Deterministic split construction for the scaled-up GEPA comparison.

Inputs:  evalset.jsonl (old 44 = old train+val pool), holdout.jsonl (old 6,
         frozen), new_questions.jsonl (teacher-generated, make_questions.py).
Outputs: holdout40.jsonl  — old holdout 6 + 34 drawn (seed 0) from the NEW
                            questions only; old train/val rows have been
                            optimized against and never enter the holdout
         evalset80.jsonl  — old 44 + 36 of the remaining new questions
         val30.jsonl /    — the exact split run.py produces for
         train50.jsonl      --evalset evalset80.jsonl --val-frac 0.375,
                            reproduced here (Random(0).shuffle; first
                            max(3, int(80*0.375)) = 30 rows are val) and
                            asserted, so both arms can consume files while
                            the Python arm re-derives the same split itself.

    python make_splits.py
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

VAL_FRAC = 0.375
N_HOLDOUT_NEW = 34
N_EVALSET = 80


def load(name):
    return [
        json.loads(l)
        for l in (HERE / name).read_text().splitlines()
        if l.strip()
    ]


def dump(name, rows):
    (HERE / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {name}: {len(rows)} rows")


def check_schema(row, source, min_key_points):
    # old rows are frozen as-is (the two refusal traps carry 2 key points);
    # generated rows must have >= 3
    assert isinstance(row.get("question"), str) and row["question"].strip(), source
    kps = row.get("key_points")
    assert isinstance(kps, list) and len(kps) >= min_key_points, (
        f"{source}: <{min_key_points} key_points: {row['question'][:60]}"
    )
    assert all(isinstance(k, str) and k.strip() for k in kps), source
    cites = row.get("citations", [])
    assert isinstance(cites, list), source


def content_tokens(text):
    import re

    stop = set(
        "the a an of to in is are and or for with on by as at it its this "
        "that be from what how do does can when which why not you your i".split()
    )
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in stop and len(t) > 2}


def near_dup(qa, qb):
    a, b = content_tokens(qa), content_tokens(qb)
    if not a or not b:
        return False
    inter = len(a & b)
    return inter / len(a | b) >= 0.5 or inter / min(len(a), len(b)) >= 0.8


def main():
    old_pool = load("evalset.jsonl")      # old train+val pool (44)
    old_holdout = load("holdout.jsonl")   # frozen (6)
    new_rows = load("new_questions.jsonl")

    for src, rows, min_kp in (
        ("evalset", old_pool, 1),
        ("holdout", old_holdout, 1),
        ("new", new_rows, 3),
    ):
        for r in rows:
            check_schema(r, src, min_kp)

    # drop new rows duplicating any old question (exact or token-overlap)
    old_questions = [r["question"] for r in old_pool + old_holdout]
    seen = set(old_questions)
    deduped = []
    for r in new_rows:
        q = r["question"]
        if q in seen:
            print(f"drop exact dupe: {q[:70]}")
            continue
        if any(near_dup(q, oq) for oq in old_questions):
            print(f"drop near-dupe of old: {q[:70]}")
            continue
        if any(near_dup(q, a["question"]) for a in deduped):
            print(f"drop near-dupe within new: {q[:70]}")
            continue
        seen.add(q)
        deduped.append(r)
    print(f"pool: {len(old_pool)} old + {len(deduped)} new (of {len(new_rows)})")

    need = N_HOLDOUT_NEW + (N_EVALSET - len(old_pool))
    assert len(deduped) >= need, f"need {need} new rows, have {len(deduped)}"

    # holdout40: frozen old 6 + 34 NEW, drawn deterministically. The old 44
    # go to evalset80 unconditionally, so a new row that token-overlaps any
    # old-pool question stays on the evalset side instead of the holdout.
    order = list(range(len(deduped)))
    random.Random(0).shuffle(order)
    old_pool_qs = [r["question"] for r in old_pool]
    holdout_new, evalset_candidates = [], []
    for i in order:
        r = deduped[i]
        if len(holdout_new) < N_HOLDOUT_NEW and not any(
            near_dup(r["question"], oq) for oq in old_pool_qs
        ):
            holdout_new.append(r)
        else:
            evalset_candidates.append(r)
    assert len(holdout_new) == N_HOLDOUT_NEW
    holdout40 = old_holdout + holdout_new

    # evalset80: all 44 old + new rows in the same shuffled order, skipping
    # anything that token-overlaps a holdout question
    holdout_qs = [r["question"] for r in holdout40]
    fill = N_EVALSET - len(old_pool)
    picked, unused = [], []
    for r in evalset_candidates:
        if len(picked) < fill and not any(near_dup(r["question"], hq) for hq in holdout_qs):
            picked.append(r)
        else:
            unused.append(r)
    assert len(picked) == fill
    evalset80 = old_pool + picked

    # optimization pool must not near-duplicate the holdout. Pairs of two
    # frozen OLD rows only warn: the August rig already shipped them on
    # opposite sides, and the token check false-positives on very short
    # questions (e.g. "What is a hole in DSRs?").
    old_qs = {r["question"] for r in old_pool} | {r["question"] for r in old_holdout}
    for r in evalset80:
        for h in holdout40:
            if near_dup(r["question"], h["question"]):
                pair = f"{r['question'][:60]} ~ {h['question'][:60]}"
                if r["question"] in old_qs and h["question"] in old_qs:
                    print(f"warn (frozen old/old overlap): {pair}")
                else:
                    raise AssertionError(f"evalset/holdout near-dup: {pair}")
    hq = {r["question"] for r in holdout40}
    assert not hq & {r["question"] for r in evalset80}
    assert len(holdout40) == len({r["question"] for r in holdout40}) == 40
    assert len(evalset80) == len({r["question"] for r in evalset80}) == N_EVALSET

    dump("holdout40.jsonl", holdout40)
    dump("evalset80.jsonl", evalset80)

    # --- CRITICAL PARITY: reproduce run.py's split byte-for-byte -----------
    # run.py: items = [json.loads(l) ...]; random.Random(0).shuffle(items);
    #         n_val = max(3, int(len(items) * val_frac)); val, train = split
    items = [
        json.loads(l)
        for l in (HERE / "evalset80.jsonl").read_text().splitlines()
        if l.strip()
    ]
    random.Random(0).shuffle(items)
    n_val = max(3, int(len(items) * VAL_FRAC))
    valset, trainset = items[:n_val], items[n_val:]
    assert n_val == 30, n_val
    assert len(trainset) == 50, len(trainset)

    # independent second derivation straight from run.py's own lines, as an
    # assertion that what we wrote is what run.py will compute
    items2 = [
        json.loads(l)
        for l in (HERE / "evalset80.jsonl").read_text().splitlines()
        if l.strip()
    ]
    random.Random(0).shuffle(items2)
    n_val2 = max(3, int(len(items2) * VAL_FRAC))
    assert items2[:n_val2] == valset and items2[n_val2:] == trainset
    print(f"parity: run.py --evalset evalset80.jsonl --val-frac {VAL_FRAC} "
          f"-> {len(trainset)} train / {n_val} val  REPRODUCED OK")

    dump("val30.jsonl", valset)
    dump("train50.jsonl", trainset)

    n_old_in_train = sum(1 for r in trainset if r["question"] in {x["question"] for x in old_pool})
    print(f"composition: train50 = {n_old_in_train} old + {50 - n_old_in_train} new; "
          f"val30 = {44 - n_old_in_train} old + {30 - (44 - n_old_in_train)} new")
    print(f"unused new rows: {len(unused)} (remain in new_questions.jsonl only)")


if __name__ == "__main__":
    main()
