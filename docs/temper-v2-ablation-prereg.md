# TEMPER v2 ablation — pre-registration (FROZEN 2026-08-25, before implementation)

Written after 5 of 9 showdown cells, before the last 4 report, and before any
v2 code exists. Tests the diagnosis that TEMPER's losses come from its search
throttles (hard length cap, always-on lexicase), not its selection machinery.

## Arms (both new; run on PUPA/toast, the cell where the cap demonstrably bound)

- `temper_postcap` — TEMPER preset, two changes: length cap applies only
  after the first promotion (bootstrap explores freely); lexicase parent
  selection also enabled only after the first promotion (Pareto before).
- `hybrid_ablate` — HYBRID (RATCHET search + variant-tempered champion)
  plus a clause-ablation finale: race single-clause removals of the champion
  on fresh∪variant columns; accept removals that don't lose (within z·SE);
  bounded rollout budget. Compactness earned causally, no hard cap.

Same budget (600), seed (0), teacher (terra), splits, judge as the campaign.
Eval: ONE fresh session re-running all five original arms plus the two new
ones, devholdout 60×2, paired, ref = python.

## Predictions (frozen)

- **P1**: `temper_postcap` closes most of TEMPER's PUPA/toast gap to RATCHET
  (composite within 0.04 of ratchet-in-same-session, vs −0.11 today). If it
  still trails ratchet by >0.06, the cap was NOT the binding constraint and
  the lexicase schedule becomes prime suspect.
- **P2**: `hybrid_ablate` matches hybrid's composite (±0.04) with a winner
  ≤60% of hybrid's length, and never below seed.
- **P3**: both new arms ship 0 train-PII leaks and 0 target-snippet leaks.
- **P4**: neither arm demotes to seed on this cell (real headroom exists:
  every arm beat seed here).

## Deviations

- 2026-08-25 ~13:00 UTC — `hybrid_ablate` v2.0 run recorded as-is: it
  shipped the seed at 402/600 rollouts, finale never fired. Diagnosis: the
  unconditional 150-rollout ablation reserve starved the search from
  generation 0 (25% less exploration than HYBRID). **P4 fails for v2.0 by
  its own mechanism, not the diagnosis under test.** v2.1 gates the reserve
  on first promotion and defaults it to 90; the rerun (same seed/budget) is
  the arm the joint eval scores. Both runs and both winner files kept.

- 2026-08-25 ~21:20 UTC — v2.1 rerun shipped a real gen-5 child (full
  budget) but the finale still never fired: rewrite fan-out lets one
  generation run two full races, overshooting `worst_generation` and eating
  the finale slack (engine's known champion under-reserve gotcha, same
  mechanism). v2.2 adds a fan-out margin to the reserve, gated behind
  `ablate_champion` so campaign arms stay byte-identical. The joint eval
  scores v2.2; all three runs kept.

- 2026-08-25 ~23:40 UTC — v2.3 (paired-race margin) left 97 rollouts for the
  finale: enough to test ONE clause of ~7 (it didn't survive; no removals).
  Ruling: **P2 is NOT TESTED** — the in-loop finale cannot get a fair budget
  share under a 600-rollout cap until the engine has a true per-generation
  worst-case model (parked as engine work; three margin attempts recorded).
  The joint eval uses the v2.3 winner, which is effectively a HYBRID
  replicate. `cargo test` green throughout; campaign arms byte-unaffected.

## Verdicts (2026-08-26, eval_toast_v2_results.jsonl, one fresh 7-arm session)

- **P1: FAIL.** temper_postcap .620 vs temper .621 (both −.12 sig vs python;
  ratchet −.019 n.s. same session). Scheduling the cap/lexicase changed
  nothing — the binding constraint is the 1500-char ceiling itself: leakage
  tracks winner length monotonically across all 7 arms (10K→.39, ~400→.71).
  On content-compounding tasks the cap excludes the winning region at any
  schedule. The un-capped HYBRID is the correct "TEMPER v2" on such tasks.
- **P2: NOT TESTED** (finale never engaged); the reserve's cost, however, is
  now measured: hybrid_ablate .601 vs hybrid .696 — throttled search, no
  finale benefit.
- **P3: PASS** — zero leaks in both new arms.
- **P4:** v2.0 failed (its own reserve bug); the evaluated v2.3 shipped a
  real child.
- Side finding: temper_postcap had the only significant QUALITY win vs
  python (+.075) and the best consistency (.950) — compact prompts answer
  better; they just cannot enumerate redaction cases.
- Session note: this session scored every repeated arm ~.04–.06 below the
  first eval session (drift) — only within-session paired deltas are valid,
  as the campaign protocol already requires.

## Not being tested

No change to the 9-cell showdown; its arms and decision rule stay as
pre-registered. This ablation informs what DSRs ships as "TEMPER v2" only if
the showdown's pooled verdict makes HYBRID/TEMPER the shipping family.
