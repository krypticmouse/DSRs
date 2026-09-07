#!/usr/bin/env python3
"""Tier-3 verdict statistics for the paired multi-arm evals (stdlib only).

Protocol (docs/reviews-2026-09-01/L3.md §1.2, §2.5, §2.7, §5.4, §6, §7 item 4)
------------------------------------------------------------------------------
Every verdict is a within-session paired design (§2.5, R3): seed and candidate
arms answer the same questions, interleaved, sharing the tool cache. The unit
of analysis is the per-question paired delta

    d_q = mean_over_passes(arm_q) - mean_over_passes(seed_q)

(transport rows excluded before pooling). Three readings of the same deltas
are printed side by side so old and new verdict logs stay comparable:

1. Percentile paired bootstrap 95% CI (10 000 resamples, rng seed 0) — the
   legacy verdict, byte-for-byte the `boot_ci` of benchmarks/*/common.py.
2. Sign-flip permutation test (§1.2): under H0 the arm labels are exchangeable
   within a question, so each d_q is symmetric about 0; the reference test is
   the Monte-Carlo (or exact, for n <= 14) distribution of the mean under
   random sign flips. No CLT, no rank transform (Wilcoxon loses power on the
   85%-null / 15%-big mixture, L3 §1.2). One-sided "greater" p is the Tier-3
   endpoint (§5.1 uses one-sided alpha).
3. Pre-registered CUPED / ANCOVA on the seed score (§7 item 4, power in
   Tier 3 of §4): covariate c_q = seed arm's per-question mean in the same
   session; theta = cov(d, c) / var(c); adjusted_q = d_q - theta (c_q - mean c).
   The point estimate is unchanged (the adjustment is mean-zero); only the
   variance shrinks, by rho^2 = corr(d, c)^2. CUPED CI = bootstrap on the
   adjusted deltas (theta held fixed) and a t test with residual variance on
   n-2 dof (identical to L3/power_sim.py::cuped_t). Legitimacy rests on the
   covariate and formula being fixed BEFORE the look — do not swap covariates
   after seeing the numbers (R2, R5).

Multiplicity (§5.4): Tier-3 looks draw from an online alpha-wealth ledger
(Foster & Stine 2008 generalized alpha-investing; LORD++ schedule of Ramdas,
Yang, Wainwright & Jordan 2017). Initial wealth W0 = .05; look j spends
alpha_j = gamma_j W0 plus, for every earlier rejection at look tau, an
earn-back stream gamma_(j - tau) * reward; gamma_j proportional to
1 / (j log^2(j + 1)), normalised to sum to one. reward = alpha - W0 for the
first rejection and alpha afterwards (LORD++; with W0 = alpha the first
rejection therefore earns 0 — pass w0 < alpha for the usual split). Every
look, pass or fail, is appended to a JSONL ledger with its alpha_j and
wealth (R4). Tiers 0-2 make no claims and do not touch the ledger.

Usage
-----
  python3 benchmarks/verdict_stats.py <eval_rows.jsonl> [--seed-arm seed]
      [--split-filter devholdout] [--metric auto|score|correct|<field>]
      [--vs ARM_A ARM_B] [--covariate ARM] [--n-boot 10000] [--n-perm 20000]
      [--ledger path/to/ledger.jsonl [--record LOOK_ID --arm ARM
       [--benchmark NAME] [--hypothesis TEXT] [--alpha .05] [--w0 .05]]]

  * rows come from any of benchmarks/{docsqa,pupa,hover,bcplus}/eval_holdout.py
    (fields arm, idx, pass|rep, score, [correct], [transport], [split], ...);
  * --metric auto picks `correct` (strict accuracy) when rows carry the
    bcplus/hover gold fields or ifbench's n_constraints, otherwise `score`;
  * every non-seed arm is compared with the seed; --vs compares two arms
    directly. CUPED covariate = the reference arm's per-question mean (seed,
    or ARM_B for --vs) unless --covariate names another arm;
  * an arm named `seed2` (a second copy of the seed prompt, interleaved in
    the same session) is the identical-prompt floor: its delta vs seed is
    printed FIRST, and every other arm's line ends with `clears-floor`
    (|delta_arm| > |delta_floor|) or `within-floor` — a delta that does not
    clear the floor is indistinguishable from session noise (2026-09-02:
    two byte-identical bcplus prompts scored .383 vs .475 at 60x2);
  * --ledger prints wealth and the next alpha_j; add --record to append the
    named arm's one-sided permutation p as a Tier-3 look.
  Library use: from verdict_stats import compare_paired, print_verdict_table.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# 1. Legacy paired bootstrap (identical to benchmarks/*/common.py::boot_ci)
# --------------------------------------------------------------------------


def paired_bootstrap(deltas, n=10_000, seed=0):
    """Percentile bootstrap 95% CI of the mean paired delta.

    Same sampler, seed and percentile indices as `boot_ci` in
    benchmarks/{bcplus,pupa,hover}/common.py so the numbers stay continuous
    with the existing verdict logs (docsqa's inline variant differs only by
    one index and by sharing one rng across arms; agreement is to ~1e-3).
    Returns (mean, lo, hi).
    """
    rng = random.Random(seed)
    m = len(deltas)
    if m == 0:
        return 0.0, 0.0, 0.0
    mean = sum(deltas) / m
    means = []
    for _ in range(n):
        sample = [deltas[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    return mean, means[int(0.025 * n)], means[int(0.975 * n)]


# --------------------------------------------------------------------------
# 2. Sign-flip permutation test
# --------------------------------------------------------------------------


def signflip_permutation(deltas, n=20_000, seed=0):
    """Paired sign-flip (randomisation) test on the mean delta.

    H0: within each question the two arm labels are exchangeable, so every
    d_q has a distribution symmetric about 0 and all 2^n sign patterns are
    equally likely. Exact enumeration when 2^n <= n_perm, else Monte Carlo
    with the add-one correction (p >= 1/(n+1), never exactly 0).
    Returns (mean, p_two_sided, p_one_sided_greater).
    """
    m = len(deltas)
    if m == 0:
        return 0.0, 1.0, 1.0
    obs = sum(deltas) / m
    tol = 1e-12
    if 2 ** m <= n:
        # Exact: enumerate every sign pattern.
        ge = abs_ge = 0
        for signs in itertools.product((1, -1), repeat=m):
            s = sum(sg * d for sg, d in zip(signs, deltas)) / m
            ge += s >= obs - tol
            abs_ge += abs(s) >= abs(obs) - tol
        total = 2 ** m
        return obs, abs_ge / total, ge / total
    rng = random.Random(seed)
    ge = abs_ge = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in deltas) / m
        ge += s >= obs - tol
        abs_ge += abs(s) >= abs(obs) - tol
    return obs, (abs_ge + 1) / (n + 1), (ge + 1) / (n + 1)


# --------------------------------------------------------------------------
# 3. CUPED / ANCOVA on the seed score
# --------------------------------------------------------------------------


def cuped(deltas, covariate):
    """Control-variate adjustment of paired deltas.

    theta = cov(delta, covariate) / var(covariate)
    adjusted_q = delta_q - theta * (covariate_q - mean(covariate))

    mean(adjusted) == mean(deltas) exactly; the sample variance drops by
    rho^2 = corr(delta, covariate)^2. Returns
    (adjusted_deltas, theta, variance_reduction_fraction).
    """
    m = len(deltas)
    if m < 3 or len(covariate) != m:
        return list(deltas), 0.0, 0.0
    dbar = sum(deltas) / m
    cbar = sum(covariate) / m
    sxx = sum((c - cbar) ** 2 for c in covariate)
    sdd = sum((d - dbar) ** 2 for d in deltas)
    sxy = sum((c - cbar) * (d - dbar) for c, d in zip(covariate, deltas))
    theta = sxy / sxx if sxx > 0 else 0.0
    adjusted = [d - theta * (c - cbar) for c, d in zip(covariate, deltas)]
    var_red = (sxy * sxy) / (sxx * sdd) if sxx > 0 and sdd > 0 else 0.0
    return adjusted, theta, var_red


def _betacf(a, b, x, max_iter=300, eps=3e-14):
    """Continued fraction for the incomplete beta (Lentz; Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T_df > t) for Student's t (one-sided upper tail), stdlib only."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return p if t >= 0 else 1.0 - p


def cuped_test(deltas, covariate, n_boot=10_000, seed=0):
    """CUPED-adjusted mean, bootstrap CI (theta fixed) and t test (n-2 dof).

    Returns dict(mean, lo, hi, theta, var_red, t, p_two, p_one_greater, se).
    """
    adjusted, theta, var_red = cuped(deltas, covariate)
    m = len(adjusted)
    mean, lo, hi = paired_bootstrap(adjusted, n=n_boot, seed=seed)
    if m > 2:
        resid_var = sum((a - mean) ** 2 for a in adjusted) / (m - 2)
        se = math.sqrt(resid_var / m) if resid_var > 0 else 0.0
    else:
        se = 0.0
    if se > 0:
        t = mean / se
        p_one = t_sf(t, m - 2)
        p_two = 2.0 * t_sf(abs(t), m - 2)
    else:
        t, p_one, p_two = (float("inf") if mean > 0 else 0.0), (0.0 if mean > 0 else 1.0), (
            0.0 if mean != 0 else 1.0
        )
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "theta": theta,
        "var_red": var_red,
        "t": t,
        "se": se,
        "p_two": min(1.0, p_two),
        "p_one_greater": p_one,
        "adjusted": adjusted,
    }


# --------------------------------------------------------------------------
# 4. One call that runs all three readings on a paired (ref, arm) list
# --------------------------------------------------------------------------


def compare_paired(pairs, covariate=None, n_boot=10_000, n_perm=20_000, seed=0, label=""):
    """Run bootstrap + sign-flip + CUPED on paired per-question scores.

    pairs      : list of (ref_score, arm_score) per question (same order).
    covariate  : per-question control variate (default: the ref scores, i.e.
                 the seed arm's own per-question mean, per L3 §7 item 4).
    Returns (result_dict, one_line_summary).
    """
    ref = [a for a, _ in pairs]
    arm = [b for _, b in pairs]
    deltas = [b - a for a, b in pairs]
    cov = list(covariate) if covariate is not None else ref
    m = len(deltas)
    mean, lo, hi = paired_bootstrap(deltas, n=n_boot, seed=seed)
    _, p_two, p_one = signflip_permutation(deltas, n=n_perm, seed=seed)
    cu = cuped_test(deltas, cov, n_boot=n_boot, seed=seed)
    boot_sig = lo > 0 or hi < 0
    cuped_sig = cu["lo"] > 0 or cu["hi"] < 0
    res = {
        "label": label,
        "n": m,
        "ref_mean": sum(ref) / m if m else 0.0,
        "arm_mean": sum(arm) / m if m else 0.0,
        "delta": mean,
        "boot_lo": lo,
        "boot_hi": hi,
        "boot_sig": boot_sig,
        "perm_p_two": p_two,
        "perm_p_one_greater": p_one,
        "cuped_lo": cu["lo"],
        "cuped_hi": cu["hi"],
        "cuped_sig": cuped_sig,
        "cuped_theta": cu["theta"],
        "cuped_var_red": cu["var_red"],
        "cuped_se": cu["se"],
        "cuped_t": cu["t"],
        "cuped_p_two": cu["p_two"],
        "cuped_p_one_greater": cu["p_one_greater"],
        "boot_width": hi - lo,
        "cuped_width": cu["hi"] - cu["lo"],
    }
    width_shrink = 1.0 - res["cuped_width"] / res["boot_width"] if res["boot_width"] > 0 else 0.0
    res["ci_width_shrink"] = width_shrink
    summary = (
        f"{label:<28} n={m:<4} Δ={mean:+.4f} boot[{lo:+.4f},{hi:+.4f}] {'SIG ' if boot_sig else 'n.s.'}"
        f"  perm p1={p_one:.4f} p2={p_two:.4f}"
        f"  CUPED[{cu['lo']:+.4f},{cu['hi']:+.4f}] {'SIG ' if cuped_sig else 'n.s.'}"
        f" p1={cu['p_one_greater']:.4f} θ={cu['theta']:+.2f} varred={100 * cu['var_red']:.0f}%"
        f" ciw={-100 * width_shrink:+.0f}%"
    )
    return res, summary


# --------------------------------------------------------------------------
# 5. Alpha-wealth ledger (LORD++ schedule, generalized alpha-investing)
# --------------------------------------------------------------------------

_GAMMA_NORM = None


def _gamma_norm(horizon=200_000):
    """1 / sum_j 1/(j log^2(j+1)); partial sum plus the integral tail
    int_N^inf dx / (x log^2 x) = 1 / log N."""
    global _GAMMA_NORM
    if _GAMMA_NORM is None:
        s = sum(1.0 / (j * math.log(j + 1) ** 2) for j in range(1, horizon + 1))
        s += 1.0 / math.log(horizon + 1)
        _GAMMA_NORM = 1.0 / s
    return _GAMMA_NORM


def gamma(j):
    """LORD++ spending sequence, gamma_j ∝ 1/(j log^2(j+1)), sum_j gamma_j = 1."""
    if j < 1:
        return 0.0
    return _gamma_norm() / (j * math.log(j + 1) ** 2)


LEDGER_FIELDS = (
    "look_id",
    "date",
    "benchmark",
    "split",
    "arm",
    "hypothesis",
    "alpha_spent",
    "p_value",
    "rejected",
    "wealth_before",
    "wealth_after",
)


def load_ledger(path):
    """Read the JSONL ledger (missing file -> empty list). Order = look order."""
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_ledger(path, entry):
    """Append one look to the JSONL ledger (creates the file)."""
    row = {k: entry.get(k) for k in LEDGER_FIELDS}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def next_alpha(ledger, alpha=0.05, w0=0.05):
    """LORD++ alpha for the next look given the ledger so far.

    j = number of looks so far + 1; tau_k = 1-based index of the k-th
    rejection in the ledger.
        alpha_j = gamma_j W0 + (alpha - W0) gamma_(j - tau_1)
                  + alpha * sum_{k >= 2} gamma_(j - tau_k)
    Returns (alpha_j, wealth_before, j). wealth_before = W0 - spent so far +
    rewards so far (reward_1 = alpha - W0, reward_k = alpha for k >= 2);
    alpha_j <= wealth_before by construction because sum_j gamma_j = 1.
    """
    j = len(ledger) + 1
    taus = [i + 1 for i, e in enumerate(ledger) if e.get("rejected")]
    a = gamma(j) * w0
    for k, tau in enumerate(taus, start=1):
        reward = (alpha - w0) if k == 1 else alpha
        a += reward * gamma(j - tau)
    spent = sum(float(e.get("alpha_spent", 0.0)) for e in ledger)
    earned = sum((alpha - w0) if k == 1 else alpha for k in range(1, len(taus) + 1))
    wealth_before = w0 - spent + earned
    a = max(0.0, min(a, wealth_before))
    return a, wealth_before, j


def record_look(path, look_id, benchmark, split, arm, hypothesis, p_value,
                alpha=0.05, w0=0.05, date=None):
    """Spend the next alpha_j on `p_value`, append to the ledger, return the row."""
    ledger = load_ledger(path)
    a_j, wealth_before, j = next_alpha(ledger, alpha=alpha, w0=w0)
    rejected = p_value <= a_j
    k = sum(1 for e in ledger if e.get("rejected")) + 1
    reward = ((alpha - w0) if k == 1 else alpha) if rejected else 0.0
    entry = {
        "look_id": look_id,
        "date": date or _dt.date.today().isoformat(),
        "benchmark": benchmark,
        "split": split,
        "arm": arm,
        "hypothesis": hypothesis,
        "alpha_spent": a_j,
        "p_value": p_value,
        "rejected": bool(rejected),
        "wealth_before": wealth_before,
        "wealth_after": wealth_before - a_j + reward,
        "look_index": j,
    }
    append_ledger(path, entry)
    return entry


# --------------------------------------------------------------------------
# 6. Row loading (all four eval_holdout.py schemas)
# --------------------------------------------------------------------------


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_transport(row):
    """docsqa tags ``transport: true`` (legacy rows: feedback 'ERROR ...');
    bcplus/pupa/hover drop transport rows before writing (rows tagged
    ``failed: true`` are NOT transport: the rollout ran and legitimately
    scored 0 — budget exhausted, empty answer — and they stay in)."""
    return bool(row.get("transport")) or str(row.get("feedback", "")).startswith("ERROR")


def detect_metric(rows):
    """'correct' for bcplus/hover/ifbench (strict accuracy is their primary;
    `score` there is partial credit / frac_found / frac_pass), else 'score'
    (docsqa judge score, pupa composite)."""
    for r in rows[:50]:
        if "found_gold" in r or "n_gold" in r or "n_constraints" in r:
            return "correct"
    return "score"


def per_question_means(rows, metric="auto", split_filter=None):
    """Pool passes into per-question means: {arm: {idx: mean}}.

    Returns (means, info) where info has metric, transport counts per arm,
    passes per question and the split filter actually applied.
    """
    if metric == "auto":
        metric = detect_metric(rows)
    has_split = any("split" in r for r in rows[:50])
    if split_filter and not has_split:
        print(f"(rows carry no `split` field; --split-filter {split_filter} not applied, kept as a label)")
        split_filter = None
    cells = defaultdict(lambda: defaultdict(list))
    transport = defaultdict(int)
    for r in rows:
        if split_filter and r.get("split") != split_filter:
            continue
        arm = r["arm"]
        if is_transport(r):
            transport[arm] += 1
            continue
        if metric not in r:
            raise KeyError(f"row lacks metric field {metric!r}: keys={sorted(r)}")
        cells[arm][r["idx"]].append(float(r[metric]))
    means = {arm: {i: sum(v) / len(v) for i, v in by_idx.items()} for arm, by_idx in cells.items()}
    passes = {arm: sorted({len(v) for v in by_idx.values()}) for arm, by_idx in cells.items()}
    info = {"metric": metric, "transport": dict(transport), "passes": passes, "split": split_filter}
    return means, info


def paired_scores(means, ref_arm, arm):
    """[(idx, ref, arm)] on the questions both arms answered."""
    shared = sorted(set(means[ref_arm]) & set(means[arm]))
    return [(i, means[ref_arm][i], means[arm][i]) for i in shared]


# --------------------------------------------------------------------------
# 7. Table printer (used by the CLI and, optionally, by eval_holdout.py)
# --------------------------------------------------------------------------


FLOOR_ARM = "seed2"  # second copy of the seed prompt = identical-prompt floor


def print_verdict_table(path, seed_arm="seed", split_filter=None, metric="auto", vs=None,
                        covariate_arm=None, n_boot=10_000, n_perm=20_000, seed=0,
                        out=sys.stdout):
    """Print bootstrap + permutation + CUPED for every non-seed arm vs seed
    (and for --vs pairs). When an arm named `seed2` is present its delta vs
    seed is printed first as the identical-prompt floor and every other arm
    is flagged clears-floor / within-floor. Returns {label: result_dict}."""
    rows = load_rows(path)
    means, info = per_question_means(rows, metric=metric, split_filter=split_filter)
    arms = list(means)
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"\n== verdict_stats: {os.path.basename(path)}  metric={info['metric']}"
      f"{'  split=' + info['split'] if info['split'] else ''} ==")
    for arm in arms:
        ks = info["passes"][arm]
        k = ks[0] if len(ks) == 1 else f"{ks[0]}..{ks[-1]}"
        p(f"  {arm:<28} q={len(means[arm]):<4} passes/q={k:<6} transport-excluded={info['transport'].get(arm, 0)}"
          f"  mean={sum(means[arm].values()) / max(1, len(means[arm])):.4f}")
    results = {}

    def run(ref, arm, tag=None, floor=None):
        # Covariate = the reference arm's own per-question mean (seed for the
        # primary endpoint; ARM_B for --vs) unless --covariate names another arm.
        # Tier-0 replay on docsqa: the seed score is useless as a covariate for
        # arm-vs-arm (theta ~ 0) while the reference arm's score cuts variance
        # 17-29%, so the uniform rule is the pre-registered one.
        cov_arm = covariate_arm or ref
        pairs = paired_scores(means, ref, arm)
        if not pairs:
            p(f"  {arm} vs {ref}: no shared questions")
            return
        cov = None
        if cov_arm != ref and cov_arm in means:
            covm = means[cov_arm]
            keep = [(i, a, b) for i, a, b in pairs if i in covm]
            if len(keep) < len(pairs):
                p(f"  (covariate arm {cov_arm} missing {len(pairs) - len(keep)} questions; paired on {len(keep)})")
            pairs = keep
            cov = [covm[i] for i, _, _ in pairs]
        label = f"{arm} − {ref}"
        res, line = compare_paired([(a, b) for _, a, b in pairs], covariate=cov,
                                   n_boot=n_boot, n_perm=n_perm, seed=seed, label=label)
        res["covariate"] = cov_arm if cov is not None else ref
        results[label] = res
        if tag:
            line += f"  <- {tag}"
        if floor is not None:
            # Identical-prompt floor: |delta| must exceed |seed2 - seed| to be
            # read as a prompt effect rather than session noise.
            res["floor_delta"] = floor["delta"]
            res["clears_floor"] = abs(res["delta"]) > abs(floor["delta"])
            line += "  clears-floor" if res["clears_floor"] else "  within-floor"
        p("  " + line)
        return res

    if seed_arm in means:
        p(f"\n  -- every arm vs {seed_arm} (covariate = {covariate_arm or seed_arm} per-question mean) --")
        floor = None
        if FLOOR_ARM in means and FLOOR_ARM != seed_arm:
            floor = run(seed_arm, FLOOR_ARM, tag="identical-prompt floor")
            if floor is not None:
                p(f"  (floor |Δ|={abs(floor['delta']):.4f}: an arm clears-floor when |Δ_arm| > |Δ_floor|,"
                  f" else within-floor = not distinguishable from same-prompt session noise)")
        for arm in arms:
            if arm != seed_arm and not (arm == FLOOR_ARM and floor is not None):
                run(seed_arm, arm, floor=floor)
    else:
        p(f"\n  (no arm named {seed_arm!r}; arms = {arms})")
    if vs:
        a, b = vs
        if a in means and b in means:
            p(f"\n  -- direct: {a} vs {b} (covariate = {covariate_arm or b} per-question mean) --")
            run(b, a)
        else:
            p(f"\n  (--vs arms not found: {vs}; arms = {arms})")
    p("  legend: boot = paired percentile bootstrap 95% CI; perm p1/p2 = sign-flip one-sided(greater)/two-sided;"
      " CUPED = adjusted-delta bootstrap CI, p1 = t test (n−2 dof); θ = cov(Δ,cov)/var(cov);"
      " varred = 1 − var(adj)/var(Δ); ciw = CUPED CI width relative to boot (negative = narrower)")
    return results


# --------------------------------------------------------------------------
# 8. CLI
# --------------------------------------------------------------------------


def _selftest():
    """Cheap invariants: t tail, exact-vs-MC permutation, CUPED mean identity, gamma sum."""
    assert abs(t_sf(1.96, 10_000) - 0.025) < 5e-4, t_sf(1.96, 10_000)
    assert abs(t_sf(2.228, 10) - 0.025) < 5e-4, t_sf(2.228, 10)
    rng = random.Random(1)
    d = [rng.gauss(0.05, 0.2) for _ in range(12)]
    m1, p2a, p1a = signflip_permutation(d, n=20_000)
    m2, p2b, p1b = signflip_permutation(d, n=2_000)
    assert abs(p1a - p1b) < 0.05 and abs(p2a - p2b) < 0.05
    c = [rng.random() for _ in d]
    adj, theta, vr = cuped(d, c)
    assert abs(sum(adj) / len(adj) - sum(d) / len(d)) < 1e-12
    assert 0.0 <= vr <= 1.0
    s = sum(gamma(j) for j in range(1, 200_001)) + _gamma_norm() / math.log(200_001)
    assert abs(s - 1.0) < 1e-6, s
    led = []
    for j in range(1, 6):
        a, w, _ = next_alpha(led, alpha=0.05, w0=0.025)
        led.append({"alpha_spent": a, "rejected": j == 2})
        assert a <= w + 1e-12
    print("selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="?", help="eval_*.jsonl written by an eval_holdout.py")
    ap.add_argument("--seed-arm", default="seed")
    ap.add_argument("--split-filter", default=None, help="keep rows whose `split` equals this")
    ap.add_argument("--metric", default="auto", help="auto | score | correct | any numeric row field")
    ap.add_argument("--vs", nargs=2, metavar=("ARM_A", "ARM_B"), help="direct ARM_A − ARM_B comparison")
    ap.add_argument("--covariate", default=None, help="arm whose per-question mean is the CUPED covariate")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=20_000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--ledger", default=None, help="alpha-wealth JSONL ledger path")
    ap.add_argument("--record", default=None, metavar="LOOK_ID", help="append a Tier-3 look for --arm to --ledger")
    ap.add_argument("--arm", default=None, help="arm recorded by --record (vs seed)")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--hypothesis", default=None)
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR target / earn-back")
    ap.add_argument("--w0", type=float, default=0.05, help="initial wealth")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        if not args.rows:
            return 0
    if not args.rows:
        ap.error("rows file required")

    results = print_verdict_table(
        args.rows, seed_arm=args.seed_arm, split_filter=args.split_filter, metric=args.metric,
        vs=tuple(args.vs) if args.vs else None, covariate_arm=args.covariate,
        n_boot=args.n_boot, n_perm=args.n_perm, seed=args.rng_seed,
    )

    if args.ledger:
        ledger = load_ledger(args.ledger)
        a_j, wealth, j = next_alpha(ledger, alpha=args.alpha, w0=args.w0)
        print(f"\n== alpha-wealth ledger {args.ledger}: {len(ledger)} looks,"
              f" {sum(1 for e in ledger if e.get('rejected'))} rejections,"
              f" wealth={wealth:.4f}, next look j={j} may spend alpha_j={a_j:.4f} ==")
        for label, r in results.items():
            print(f"  {label:<28} one-sided perm p={r['perm_p_one_greater']:.4f} ->"
                  f" {'REJECT' if r['perm_p_one_greater'] <= a_j else 'fail'} at alpha_j={a_j:.4f}")
        if args.record:
            if not args.arm:
                ap.error("--record needs --arm")
            label = f"{args.arm} − {args.seed_arm}"
            if label not in results:
                ap.error(f"{label} not in results")
            r = results[label]
            bench = args.benchmark or os.path.basename(os.path.dirname(os.path.abspath(args.rows)))
            entry = record_look(
                args.ledger, args.record, bench, args.split_filter or "", args.arm,
                args.hypothesis or f"{args.arm} > {args.seed_arm} (one-sided sign-flip, CUPED pre-registered)",
                r["perm_p_one_greater"], alpha=args.alpha, w0=args.w0,
            )
            print("  recorded:", json.dumps(entry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
