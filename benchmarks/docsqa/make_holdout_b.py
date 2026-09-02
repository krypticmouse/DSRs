"""Mint DocsQA holdout-B — the pre-sealed replacement for holdout100.

Why: docs/reviews-2026-09-01/L3.md §5.4 / R12 retires holdout100 after 30
Tier-3 looks (~22 spent as of 2026-09-02) and requires a pre-sealed
replacement. Both question pools are fully allocated (pool_existing 147 +
pool_generated 189 = every row of holdout100/devholdout/train/val/reserve;
pool_certify 278 = certify150, consumed by the optimizer's certification
step), so holdout-B must be freshly generated.

Template: make_certify_dataset.py (page walk, prompts, dedupe, lexical +
independent-LLM verification, same teacher/verifier models and
temperatures) with these deliberate changes:

  1. Dedupe against EVERY existing row — pool_existing, pool_generated,
     pool_certify AND all six split files — by id and by make_dataset's
     question near-duplicate test (Jaccard >= .5 or containment >= .8 on
     content tokens).
  2. Fresh ids `hbNNN` — id-disjoint from ex/gen/cert by construction and
     verified against every file before anything under data/ is written.
  3. Cross-page pairings drawn with rng seed 2 (seed 0 = original build,
     seed 1 = certify build).
  4. Quotas sized for the tripled dedupe pressure (QUOTA MATH below).
  5. Stratified final draw matching holdout100's tier composition
     (untiered legacy `ex` rows are mapped by citation count: 0 ->
     negative, 1 -> single_page, 2+ -> cross_page; all 50 have exactly 1).
  6. --dry-run (plan only: zero API calls, zero writes) and --concurrency
     (default 2 — gentle on a shared OpenRouter account).

Outputs (real run only; the script refuses to overwrite an existing
holdout_b.jsonl without --force, because a sealed holdout is sealed):
  data/pool_holdout_b.jsonl   full verified pool (record; serve skips pool*)
  data/holdout_b.jsonl        the sealed split: exactly --n rows (100),
                              stratified by tier, seeded shuffle seed 0;
                              its SHA256 is printed for the prereg
  rejects/hb_rejects.jsonl    rejects with reasons (audit artifact — NEVER
                              under data/: serve loads every data/*.jsonl
                              except pool*/gen_* as an eval split, and rows
                              without key_points kill it at startup — the
                              2026-08-31 forage4 verdict-1 void)

Serve picks the new split up automatically: example 36-docsqa's serve mode
read_dir()s data/, so data/holdout_b.jsonl becomes split "holdout_b".

Run with system python3 (stdlib only); the real run needs OPENROUTER_API_KEY:

    python3 make_holdout_b.py --dry-run
    python3 make_holdout_b.py --concurrency 2 | tee rejects/hb_build.log
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import make_dataset as md

HERE = Path(__file__).parent
DATA = HERE / "data"
POOL_OUT = DATA / "pool_holdout_b.jsonl"
SPLIT_OUT = DATA / "holdout_b.jsonl"
REJECTS = HERE / "rejects" / "hb_rejects.jsonl"
COMPOSITION_REF = DATA / "holdout100.jsonl"

# every file that holds an existing row: the three pools + the six splits
POOL_FILES = ["pool_existing.jsonl", "pool_generated.jsonl", "pool_certify.jsonl"]
SPLIT_FILES = ["holdout100.jsonl", "devholdout.jsonl", "train.jsonl",
               "val.jsonl", "reserve.jsonl", "certify150.jsonl"]
ALL_FILES = POOL_FILES + SPLIT_FILES

ID_PREFIX = "hb"
PAIR_SEED = 2        # seed 0 = make_dataset, seed 1 = make_certify_dataset
SHUFFLE_SEED = 0     # same final-shuffle seed as both templates

# QUOTA MATH (from the two recorded builds: rejects/gen_rejects.jsonl,
# rejects/cert_rejects.jsonl, rejects/cert_build.log):
#   build 1  make_dataset          pool 147  base quota, cross n=2 -> 189 verified
#            rejects 23 (11 near-dup, 7 topic-present, 5 verify)
#   build 2  make_certify_dataset  pool 336  quota +2,  cross n=3 -> 278 verified
#            ~190 single / 87 cross / 20 neg candidates -> 178 / 85 / 15
#            accepted (94% / 98% / 75%); rejects 19 (10 near-dup, 6 verify,
#            3 topic-present)
#   build 3  THIS                  pool 614  (1.8x build 2, 4.2x build 1; the
#            same pages have now been mined twice, so the obvious questions
#            are gone and the near-dup filter bites harder)
# Planning floor: assume near-dup climbs from ~3% to ~25% and lexical+verify
# from ~3% to ~15% -> 60% yield on single/cross (vs 94-98% observed); 50% on
# negatives (that prompt carries no existing-question list and 28 negatives
# already exist, so collisions are likelier).
#   single   quota +2 -> 196 requested x 0.60 = ~118  >= 84 target  (1.4x)
#            (base quota 134 x 0.60 = ~80 would MISS the 84 target)
#   cross    n=2 x ~33 pairs = ~66 x 0.60 = ~40         >= 10 target  (4x)
#   negative 2 calls x 20 = 40 x 0.50 = ~20             >=  6 target  (3x)
#   total    ~178 verified rows >= 100 even at the floor; ~290 at observed
#            yields. Cross n=2 (not certify's 3): cross is only 10% of the
#            target and every accepted row costs one verifier call.
SINGLE_QUOTA_BONUS = 2
CROSS_N = 2
NEG_CALLS = 2
NEG_N = 20
YIELD_FLOOR = 0.60       # post-verification yield assumed for single/cross
YIELD_FLOOR_NEG = 0.50
PREVERIFY_FLOOR = 0.70   # share of candidates that reach the verifier (floor)


def load_jsonl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def make_pairs(page_text, seed):
    """Cross-page pairings — byte-identical recipe to both templates, only
    the rng seed differs."""
    rng = random.Random(seed)
    by_dir: dict[str, list[str]] = {}
    for s in page_text:
        by_dir.setdefault(s.split("/")[0], []).append(s)
    pairs = []
    for _, slugs in sorted(by_dir.items()):
        slugs = sorted(slugs)
        rng.shuffle(slugs)
        pairs += [(slugs[i], slugs[i + 1]) for i in range(0, len(slugs) - 1, 2)]
    for _, slugs in sorted(by_dir.items()):
        slugs = sorted(slugs)
        rng.shuffle(slugs)
        pairs += [(slugs[i], slugs[i + 1]) for i in range(1, len(slugs) - 1, 2)]
    comp = sorted(by_dir.get("components", []))
    opt = sorted(by_dir.get("optimizers", []))
    gui = sorted(by_dir.get("guides", []))
    rng.shuffle(comp), rng.shuffle(opt), rng.shuffle(gui)
    pairs += list(zip(comp[:8], opt[:8]))
    pairs += list(zip(gui[:6], comp[8:14] if len(comp) > 13 else comp[:6]))
    return list(dict.fromkeys(pairs))


def effective_tier(row):
    if row.get("tier"):
        return row["tier"]
    n = len(row.get("citations") or [])
    return "negative" if n == 0 else "single_page" if n == 1 else "cross_page"


def target_counts(ref_rows, n):
    """Largest-remainder apportionment of n over the reference's tiers."""
    counts = Counter(effective_tier(r) for r in ref_rows)
    total = sum(counts.values())
    raw = {t: n * c / total for t, c in counts.items()}
    tgt = {t: int(v) for t, v in raw.items()}
    for t, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]),
                       reverse=True)[: n - sum(tgt.values())]:
        tgt[t] += 1
    return counts, tgt


def draw_split(final, target):
    """Stratified draw: the first target[tier] rows of each tier in the
    pool's seeded order; shortfalls backfilled single_page -> cross_page ->
    negative; then one seeded shuffle of the drawn rows."""
    by_tier: dict[str, list[dict]] = {}
    for r in final:
        by_tier.setdefault(r["tier"], []).append(r)
    chosen, short = [], {}
    for t, k in target.items():
        avail = by_tier.get(t, [])
        chosen += avail[:k]
        if len(avail) < k:
            short[t] = k - len(avail)
    need = sum(short.values())
    used = {r["id"] for r in chosen}
    for t in ("single_page", "cross_page", "negative"):
        for r in by_tier.get(t, []):
            if need == 0:
                break
            if r["id"] not in used:
                chosen.append(r)
                used.add(r["id"])
                need -= 1
    random.Random(SHUFFLE_SEED).shuffle(chosen)
    return chosen, short


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (pages, quotas, pairs, expected calls); no API calls, no writes")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel LLM calls (default 2 — the OpenRouter account is shared)")
    ap.add_argument("--n", type=int, default=100, help="rows in holdout_b.jsonl (default 100)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing data/holdout_b.jsonl (a sealed holdout!)")
    args = ap.parse_args()

    # -- dedupe pool: every row of every file, by id and by question
    per_file = {name: load_jsonl(DATA / name) for name in ALL_FILES}
    all_ids: set[str] = set()
    all_questions: list[str] = []
    seen_q: set[str] = set()
    for name in ALL_FILES:
        for r in per_file[name]:
            if r.get("id"):
                all_ids.add(r["id"])
            if r["question"] not in seen_q:
                seen_q.add(r["question"])
                all_questions.append(r["question"])
    existing_block = "\n".join(f"- {q}" for q in all_questions)
    n_rows = sum(len(v) for v in per_file.values())
    print("dedupe pool: " + ", ".join(f"{k[:-6]} {len(v)}" for k, v in per_file.items()))
    print(f"dedupe pool: {n_rows} rows across {len(ALL_FILES)} files -> "
          f"{len(all_questions)} unique questions, {len(all_ids)} unique ids "
          f"(prefixes {sorted({''.join(c for c in i if not c.isdigit()) for i in all_ids})})")
    hb_clash = sorted(i for i in all_ids if i.startswith(ID_PREFIX))
    print(f"existing ids with prefix '{ID_PREFIX}': {hb_clash or 'NONE'}")

    # -- tier target from holdout100's composition
    ref = load_jsonl(COMPOSITION_REF)
    raw_tiers = Counter((r.get("tier") or "untiered(legacy ex)") for r in ref)
    raw_origin = Counter(r.get("origin") for r in ref)
    eff_counts, target = target_counts(ref, args.n)
    print(f"{COMPOSITION_REF.name}: {len(ref)} rows; tier {dict(raw_tiers)}; origin {dict(raw_origin)}")
    print(f"  effective tiers (legacy rows mapped by citation count): {dict(eff_counts)}")
    print(f"  holdout_b target ({args.n} rows): {target}")

    # -- pages + fresh pairs
    pages = list(md.iter_pages())
    page_text = {md.slug_of(p): p.read_text(encoding="utf-8") for p in pages}
    page_tok = {s: set(md.tokens(t)) for s, t in page_text.items()}
    quota = {md.slug_of(p): md.questions_for(p) + SINGLE_QUOTA_BONUS for p in pages}
    pairs = make_pairs(page_text, PAIR_SEED)
    mined = {frozenset(r["source_pages"]) for name in POOL_FILES for r in per_file[name]
             if r.get("tier") == "cross_page" and r.get("source_pages")}
    n_mined = sum(frozenset(p) in mined for p in pairs)
    print(f"{len(pages)} pages, {len(pairs)} cross-page pairs (seed {PAIR_SEED}); "
          f"{n_mined} of the pairs were also mined by an earlier build (question-level "
          f"dedupe still applies)")

    cand_single = sum(quota.values())
    cand_cross = CROSS_N * len(pairs)
    cand_neg = NEG_CALLS * NEG_N
    exp_single = int(cand_single * YIELD_FLOOR)
    exp_cross = int(cand_cross * YIELD_FLOOR)
    exp_neg = int(cand_neg * YIELD_FLOOR_NEG)

    if args.dry_run:
        print("\n== page plan (size bytes, single_page quota = questions_for + "
              f"{SINGLE_QUOTA_BONUS}) ==")
        for p in pages:
            s = md.slug_of(p)
            print(f"  {p.stat().st_size:6d}  n={quota[s]}  {s}")
        print(f"\n== cross-page pairs (seed {PAIR_SEED}, n={CROSS_N} each) ==")
        for a, b in pairs:
            print(f"  {'*' if frozenset((a, b)) in mined else ' '} {a} + {b}")
        print("  (* = pair also mined by an earlier build)")
        print("\n== quota math ==")
        print(f"  single_page : {len(pages)} pages, quota sum {cand_single} "
              f"(base {cand_single - SINGLE_QUOTA_BONUS * len(pages)} + {SINGLE_QUOTA_BONUS}/page)"
              f" x {YIELD_FLOOR:.2f} floor yield = ~{exp_single}  (target {target.get('single_page', 0)})")
        print(f"  cross_page  : {len(pairs)} pairs x {CROSS_N} = {cand_cross}"
              f" x {YIELD_FLOOR:.2f} = ~{exp_cross}  (target {target.get('cross_page', 0)})")
        print(f"  negative    : {NEG_CALLS} calls x {NEG_N} = {cand_neg}"
              f" x {YIELD_FLOOR_NEG:.2f} = ~{exp_neg}  (target {target.get('negative', 0)})")
        tot_floor = exp_single + exp_cross + exp_neg
        tot_obs = int(cand_single * 0.94 + cand_cross * 0.98 + cand_neg * 0.75)
        print(f"  expected verified pool: ~{tot_floor} at the floor, ~{tot_obs} at build-2 yields"
              f"  (need >= {args.n}; per-tier targets above)")
        ok = all(
            {"single_page": exp_single, "cross_page": exp_cross, "negative": exp_neg}.get(t, 0) >= k
            for t, k in target.items())
        print(f"  floor covers every tier target: {'YES' if ok else 'NO'}")
        gen_calls = len(pages) + len(pairs) + NEG_CALLS
        v_lo = int(PREVERIFY_FLOOR * (cand_single + cand_cross))
        v_hi = cand_single + cand_cross
        print("\n== expected API calls (teacher "
              f"{md.TEACHER} t=0.7, verifier {md.VERIFIER} t=0) ==")
        print(f"  generation : {len(pages)} single + {len(pairs)} cross + {NEG_CALLS} negative = {gen_calls}")
        print(f"  verification: one per non-negative candidate that survives dedupe+lexical: "
              f"~{v_lo} (floor) .. {v_hi} (max)")
        print(f"  TOTAL      : ~{gen_calls + v_lo} .. {gen_calls + v_hi} "
              f"(+ up to 2 retries per failed call, per make_dataset.llm)")
        print(f"  concurrency: {args.concurrency}")
        print(f"\n== outputs (NOT written in --dry-run) ==\n  {POOL_OUT}\n  {SPLIT_OUT}\n  {REJECTS}")
        print("dry-run: no API calls made, nothing written.")
        return

    # -- real run: guard the seal before spending a single call
    for out in (SPLIT_OUT, POOL_OUT):
        if out.exists() and not args.force:
            raise SystemExit(f"{out} already exists — it is a sealed holdout. "
                             "Re-run with --force only if you mean to re-mint it.")
    if not md.os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")

    REJECTS.parent.mkdir(exist_ok=True)
    accepted: list[dict] = []
    rejects = REJECTS.open("w")
    pool = ThreadPoolExecutor(args.concurrency)

    def consider(q, kps, tier, sources):
        if md.near_duplicate(q, all_questions):
            rejects.write(json.dumps({"reason": "near-dup", "question": q}) + "\n")
            return
        toks = set()
        for s in sources:
            toks |= page_tok[s]
        kept = [k for k in kps if md.lexical_support(k, toks)] if sources else kps
        if sources and len(kept) < 3:
            rejects.write(json.dumps({"reason": "lexical", "question": q, "kept": len(kept)}) + "\n")
            return
        accepted.append({
            "question": q, "key_points": kept, "citations": sources,
            "tier": tier, "source_pages": sources, "origin": "gen-holdout-b",
        })
        all_questions.append(q)

    # -- single-page generation (quota +2, as the certify build)
    def gen_single(path):
        slug = md.slug_of(path)
        try:
            out = md.llm(md.TEACHER, md.GEN_SINGLE.format(
                n=quota[slug], existing=existing_block,
                slug=slug, page=page_text[slug][:24000]), 0.7)
            return slug, out.get("questions", [])
        except Exception as e:  # noqa: BLE001
            print(f"  gen failed {slug}: {e}", flush=True)
            return slug, []

    for slug, qs in pool.map(gen_single, pages):
        for c in qs:
            if c.get("question") and c.get("key_points"):
                consider(c["question"], c["key_points"], "single_page", [slug])
    print(f"single_page accepted so far: {len(accepted)}", flush=True)

    # -- cross-page generation (n=2)
    def gen_cross(pair):
        a, b = pair
        try:
            out = md.llm(md.TEACHER, md.GEN_CROSS.format(
                n=CROSS_N, existing=existing_block, slug_a=a, page_a=page_text[a][:14000],
                slug_b=b, page_b=page_text[b][:14000]), 0.7)
            return pair, out.get("questions", [])
        except Exception as e:  # noqa: BLE001
            print(f"  gen failed {a}+{b}: {e}", flush=True)
            return pair, []

    n_before = len(accepted)
    for (a, b), qs in pool.map(gen_cross, pairs):
        for c in qs:
            if c.get("question") and c.get("key_points"):
                consider(c["question"], c["key_points"], "cross_page", [a, b])
    print(f"cross_page accepted: {len(accepted) - n_before}", flush=True)

    # -- negative generation (same recipe; NEG_CALLS draws because that prompt
    #    has no existing-question list and 28 negatives already exist)
    index = "\n".join(
        f"- {s}: {page_text[s].splitlines()[0][:80]}" for s in sorted(page_text)
    )
    all_docs_tokens = set().union(*page_tok.values())

    def gen_neg(_i):
        try:
            return md.llm(md.TEACHER, md.GEN_NEGATIVE.format(n=NEG_N, index=index), 0.7).get("questions", [])
        except Exception as e:  # noqa: BLE001
            print(f"  gen failed negative: {e}", flush=True)
            return []

    n_before = len(accepted)
    for qs in pool.map(gen_neg, range(NEG_CALLS)):
        for c in qs:
            q, topic = c.get("question"), [t.lower() for t in c.get("topic_tokens", [])]
            if not q or not topic:
                continue
            if sum(t in all_docs_tokens for t in topic) > len(topic) // 2:
                rejects.write(json.dumps({"reason": "topic-present", "question": q}) + "\n")
                continue
            consider(
                q,
                [
                    "The DSRs documentation does not cover this topic.",
                    "A correct answer states the docs don't cover it instead of guessing or inventing an answer.",
                ],
                "negative",
                [],
            )
    print(f"negative accepted: {len(accepted) - n_before}", flush=True)

    # -- independent LLM verification pass (identical to both templates)
    to_verify = [r for r in accepted if r["tier"] != "negative"]

    def verify(row):
        pages_blob = "\n\n---\n\n".join(
            f"[{s}]\n{page_text[s][:16000]}" for s in row["source_pages"]
        )
        points = "\n".join(f"{i+1}. {k}" for i, k in enumerate(row["key_points"]))
        try:
            out = md.llm(md.VERIFIER, md.VERIFY.format(pages=pages_blob, points=points), 0.0, 1000)
            flags = out.get("supported", [])
            kept = [k for k, ok in zip(row["key_points"], flags) if ok]
            return row, kept
        except Exception as e:  # noqa: BLE001
            print(f"  verify failed {row['question'][:50]}: {e}", flush=True)
            return row, row["key_points"]  # keep on verifier failure

    final = [r for r in accepted if r["tier"] == "negative"]
    dropped = 0
    for row, kept in pool.map(verify, to_verify):
        if len(kept) < 3:
            dropped += 1
            rejects.write(json.dumps({"reason": "verify", "question": row["question"], "kept": len(kept)}) + "\n")
            continue
        if len(kept) < len(row["key_points"]):
            row = {**row, "key_points": kept}
        final.append(row)
    pool.shutdown()
    rejects.close()
    print(f"verification dropped {dropped} rows -> {len(final)} verified "
          f"(tiers {dict(Counter(r['tier'] for r in final))}); rejects -> {REJECTS}", flush=True)

    # -- ids: seeded shuffle of the verified pool, then hbNNN in that order
    random.Random(SHUFFLE_SEED).shuffle(final)
    for i, r in enumerate(final):
        r["id"] = f"{ID_PREFIX}{i:03d}"

    # -- disjointness vs EVERY existing file, by id, exact question, near-dup
    id_clashes = [r["id"] for r in final if r["id"] in all_ids]
    q_clashes = [r["id"] for r in final if r["question"] in seen_q]
    near = [r["id"] for r in final if md.near_duplicate(r["question"], list(seen_q))]
    print(f"disjointness vs {len(ALL_FILES)} files: id clashes {id_clashes or 'NONE'}, "
          f"exact-question clashes {q_clashes or 'NONE'}, near-dup {near or 'NONE'}")
    if id_clashes or q_clashes or near:
        raise SystemExit("DISJOINTNESS VIOLATION — nothing written under data/")
    bad = [r["id"] for r in final if not r.get("question") or not r.get("key_points")]
    if bad:
        raise SystemExit(f"rows without question/key_points would kill serve: {bad}")

    with POOL_OUT.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool: {len(final)} rows -> {POOL_OUT}  sha256 {sha256_of(POOL_OUT)}")

    chosen, short = draw_split(final, target)
    if short:
        print(f"WARNING: tier shortfall {short} — backfilled from other tiers; "
              "composition will not match holdout100 exactly")
    if len(chosen) < args.n:
        raise SystemExit(f"only {len(chosen)} verified rows for a {args.n}-row split — "
                         "holdout_b.jsonl NOT written; raise the quotas and re-run")
    with SPLIT_OUT.open("w") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -- re-read what was sealed and report
    sealed = load_jsonl(SPLIT_OUT)
    tiers = dict(Counter(r["tier"] for r in sealed))
    assert len(sealed) == args.n and all(r["key_points"] for r in sealed)
    assert not ({r["id"] for r in sealed} & all_ids)
    print(f"FINAL: {len(sealed)} rows -> {SPLIT_OUT}  tiers {tiers}  target {target}")
    print(f"SHA256 {SPLIT_OUT.name}: {sha256_of(SPLIT_OUT)}   <- seal this in the prereg")


if __name__ == "__main__":
    main()
