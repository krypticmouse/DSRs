//! Standalone ship certification — the RATCHET pawl as a library API.
//!
//! Prompt optimizers have two separable halves: a generator (proposes
//! candidate prompts) and a ship decision (decides what reaches
//! production). This module is the ship decision alone, detached from any
//! generator: an anytime-valid sequential race of ANY candidate against
//! the incumbent, paired column by column on data the generator never
//! optimized against ([`PairedEvidence`] carries the statistics). Feed it
//! a winner from GEPA, RATCHET, pip-gepa, or a hand-written prompt; it
//! returns a certificate, a refusal, or a harm verdict — never a blind
//! argmax ship.
//!
//! The protocol, exactly as live-validated (11 correct verdicts across
//! three benchmarks, then the first certified gepa-class ship at e 12.01
//! — `docs/handoff-gepa-engine-ratchet-pawl.md`):
//!
//! 1. Column order is shuffled ONCE, seeded, before any scores exist.
//! 2. Each column is one paired fresh rollout: candidate and incumbent on
//!    the same example, delta = candidate − incumbent.
//! 3. Bets may warm-start from predictable past data (e.g. the
//!    generator's own validation delta); the wealth never does.
//! 4. Stop the moment gain OR harm certifies at `alpha`; report
//!    `(verdict, cols, mean, e)` always, refusals included.
//!
//! Validity is anytime (Ville), so checking after every block is free.
//! What it does NOT license: re-running a refused certification at a new
//! alpha or on re-drawn data (multiplicity — the guarantee is per
//! attempt), or certifying on columns the generator selected against.
//! The self-scaling fee is real and between-question: columns ≈
//! 2·ln(1/α)·(σ/Δ)², so budget ≥150 generator-untouched rows for
//! +.05-class effects; multi-pass columns do NOT shrink the spend
//! (measured: the variance is between-question, not sampling noise).

use anyhow::{Context, Result, bail};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::future::Future;

use crate::optimizer::engine::{BatchEvalOutcome, Candidate, Engine};
use crate::optimizer::evidence::{BandBets, PairedEvidence};
use crate::optimizer::target::OptimizeTarget;

/// A bet warm-start: `weight` pseudo-observations at `mean ± sd`, entering
/// the bet plug-in only ([`PairedEvidence::with_bet_prior`] — a wrong prior
/// costs growth, never validity). Use predictable past data only, e.g. the
/// generator's own validation estimate of the effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetPrior {
    /// Anticipated paired delta, raw score units.
    pub mean: f64,
    /// Anticipated between-column spread of the delta (~.18–.25 on 0-to-1
    /// judge metrics, measured).
    pub sd: f64,
    /// Pseudo-observation weight; ~25 washes out within a few blocks.
    pub weight: f64,
}

/// Knobs for one certification attempt. The defaults are the
/// campaign-standard preset (`alpha` .1, single-pass, block 6) — the
/// deployment pattern is ONE attempt at preset settings per candidate.
#[derive(Clone, Debug)]
pub struct CertifyConfig {
    /// Ship level: certifies when a wallet reaches `1/alpha`.
    pub alpha: f64,
    /// Score range fed to [`PairedEvidence::new`]: deltas are clamped to
    /// `[-scale, scale]`. `1.0` for 0-to-1 metrics.
    pub scale: f64,
    /// Optional bet warm-start (predictable data only). Cold-starting a
    /// small-effect certification measurably wastes the first ~20 columns
    /// of growth.
    pub prior: Option<BetPrior>,
    /// Optional fixed-bet band mixed with the aGRAPA wallet (E5a power
    /// repair): fixed Kelly bets for the anticipated effect band start
    /// growing from column 1 instead of waiting for the plug-in to warm
    /// up — the repair for the +.03-class conservative miss. Use
    /// [`BandBets::default`] (the measured .02–.06 campaign band) or
    /// [`BandBets::for_horizon`] with the attempt's pre-committed column
    /// budget. `None` = the v4 wallet, byte-identical.
    pub band: Option<BandBets>,
    /// Paired samples averaged per column before the e-process observes
    /// the delta. Keep at 1: multi-pass does not shrink the between-question
    /// variance that dominates (measured, hybrid attempt 3).
    pub passes: usize,
    /// Seed for the one up-front column-order shuffle.
    pub seed: u64,
    /// Columns evaluated concurrently per block; the wallets are checked
    /// at block boundaries (free, by anytime validity).
    pub block: usize,
    /// Hard cap on columns spent (`None` = every row supplied).
    pub max_cols: Option<usize>,
}

impl Default for CertifyConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            scale: 1.0,
            prior: None,
            band: None,
            passes: 1,
            seed: 0,
            block: 6,
            max_cols: None,
        }
    }
}

/// What one certification attempt concluded. Always fully reported —
/// a refusal's open evidence state (`e_gain`, `mean_delta`, `cols`) is the
/// honest record, and distinguishes "refused, e≈1, mean≈0" from "refused,
/// e≈6, mean≈+.05, ran out of columns".
#[derive(Clone, Copy, Debug)]
pub struct CertifyVerdict {
    /// `Some(true)` = ship the candidate (gain certified at `alpha`);
    /// `Some(false)` = ship the incumbent (harm certified); `None` =
    /// evidence ran out unproven — ship the incumbent (honest refusal).
    pub certified: Option<bool>,
    /// Paired columns consumed.
    pub cols: usize,
    /// Mean paired delta over the consumed columns, raw score units.
    pub mean_delta: f64,
    /// Final gain-wallet e-value (bar is `1/alpha`).
    pub e_gain: f64,
    /// Final harm-wallet e-value.
    pub e_harm: f64,
    /// The level the attempt ran at.
    pub alpha: f64,
}

impl CertifyVerdict {
    /// Whether the candidate earned the ship.
    pub fn ship_candidate(&self) -> bool {
        self.certified == Some(true)
    }

    fn snapshot(evidence: &PairedEvidence, certified: Option<bool>, alpha: f64) -> Self {
        Self {
            certified,
            cols: evidence.count(),
            mean_delta: evidence.mean_delta(),
            e_gain: evidence.e_gain(),
            e_harm: evidence.e_harm(),
            alpha,
        }
    }
}

impl std::fmt::Display for CertifyVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CERTIFY VERDICT: {:?} after {} cols — mean paired delta {:+.4}, e-value {:.2} (bar {:.0})",
            self.certified,
            self.cols,
            self.mean_delta,
            self.e_gain,
            1.0 / self.alpha
        )
    }
}

fn evidence_for(config: &CertifyConfig) -> PairedEvidence {
    let evidence = match config.prior {
        Some(prior) => {
            PairedEvidence::with_bet_prior(config.scale, prior.mean, prior.sd, prior.weight)
        }
        None => PairedEvidence::new(config.scale),
    };
    match &config.band {
        Some(band) => evidence.banded(band),
        None => evidence,
    }
}

/// The one up-front shuffle: fixed before any scores exist, independent of
/// content, truncated to the column budget.
fn column_order(num_rows: usize, config: &CertifyConfig) -> Vec<usize> {
    let mut order: Vec<usize> = (0..num_rows).collect();
    order.shuffle(&mut StdRng::seed_from_u64(config.seed));
    if let Some(cap) = config.max_cols {
        order.truncate(cap);
    }
    order
}

/// Certifies a candidate against an incumbent through a caller-supplied
/// paired rollout — the rig-agnostic lane for harnesses whose students
/// aren't [`OptimizeTarget`] modules (external HTTP agents, serve-mode
/// rigs, judge pipelines).
///
/// `paired_delta(row, pass)` runs ONE fresh candidate rollout and ONE
/// fresh incumbent rollout on row `row` and returns candidate − incumbent
/// in raw score units. It owns the rig's failure policy (e.g. score a
/// transport failure 0.0 for both arms); an `Err` aborts the attempt.
/// Rows within a block run concurrently; a column's `passes` samples run
/// sequentially inside it and are averaged before the e-process looks.
///
/// `on_block` fires after every block with the running (uncertified)
/// state — progress logging; pass `|_| {}` to skip.
pub async fn certify_paired<F, Fut>(
    num_rows: usize,
    config: &CertifyConfig,
    paired_delta: F,
    mut on_block: impl FnMut(&CertifyVerdict),
) -> Result<CertifyVerdict>
where
    F: Fn(usize, usize) -> Fut,
    Fut: Future<Output = Result<f64>>,
{
    if num_rows == 0 {
        bail!("certification needs at least one row");
    }
    let passes = config.passes.max(1);
    let order = column_order(num_rows, config);
    let mut evidence = evidence_for(config);
    let mut certified: Option<bool> = None;

    for block in order.chunks(config.block.max(1)) {
        let futures: Vec<_> = block
            .iter()
            .map(|&row| {
                let paired_delta = &paired_delta;
                async move {
                    let mut total = 0.0;
                    for pass in 0..passes {
                        total += paired_delta(row, pass)
                            .await
                            .with_context(|| format!("paired rollout on row {row}"))?;
                    }
                    Ok::<f64, anyhow::Error>(total / passes as f64)
                }
            })
            .collect();
        for delta in futures::future::join_all(futures).await {
            evidence.observe(delta?);
        }
        let running = CertifyVerdict::snapshot(&evidence, None, config.alpha);
        tracing::info!(
            cols = running.cols,
            mean_delta = running.mean_delta,
            e_gain = running.e_gain,
            e_harm = running.e_harm,
            "certification block"
        );
        on_block(&running);
        if evidence.certified_gain(config.alpha) {
            certified = Some(true);
            break;
        }
        if evidence.certified_harm(config.alpha) {
            certified = Some(false);
            break;
        }
    }

    let verdict = CertifyVerdict::snapshot(&evidence, certified, config.alpha);
    tracing::info!(?verdict.certified, cols = verdict.cols, mean_delta = verdict.mean_delta,
        e_gain = verdict.e_gain, "sequential ship certification");
    Ok(verdict)
}

/// Certifies a registered-candidate ship over an [`OptimizeTarget`] — the
/// module/program lane. The target's example set IS the certification
/// split, so hand it rows the generator never optimized against (reserve
/// ≥150 at dataset-construction time for +.05-class effects), and give the
/// engine a fresh rollout cache (new [`Engine`] or unused `cache_salt`):
/// cache-served rollouts would replay the generator's own draws, and the
/// protocol requires fresh ones.
///
/// Runs the identical block loop as [`certify_paired`] through
/// [`Engine::evaluate_many`], stopping early on either wallet or on
/// budget exhaustion (verdict `None` — an honest refusal, reported with
/// its open evidence state).
pub async fn certify_candidate(
    target: &OptimizeTarget<'_>,
    engine: &mut Engine,
    candidate: &Candidate,
    incumbent: &Candidate,
    config: &CertifyConfig,
) -> Result<CertifyVerdict> {
    if config.passes > 1 {
        bail!(
            "multi-pass certification is closure-lane only (certify_paired): the engine \
             caches (candidate, example) rollouts, so repeat passes replay the same draw \
             — and measured, multi-pass does not help (between-question variance)"
        );
    }
    let num_rows = target.num_examples();
    if num_rows == 0 {
        bail!("certification needs at least one row");
    }
    let candidate_row = engine.register(candidate.clone());
    let incumbent_row = engine.register(incumbent.clone());
    if candidate_row == incumbent_row {
        bail!("candidate and incumbent are identical — nothing to certify");
    }

    let order = column_order(num_rows, config);
    let mut evidence = evidence_for(config);
    let mut certified: Option<bool> = None;

    for block in order.chunks(config.block.max(1)) {
        let evals = match engine
            .evaluate_many(target, &[candidate_row, incumbent_row], Some(block))
            .await?
        {
            BatchEvalOutcome::Complete(evals) => evals,
            BatchEvalOutcome::BudgetExhausted { .. } => break,
        };
        for (cand, seed) in evals[0].rollouts.iter().zip(&evals[1].rollouts) {
            evidence.observe(cand.eval.score - seed.eval.score);
        }
        tracing::info!(
            cols = evidence.count(),
            mean_delta = evidence.mean_delta(),
            e_gain = evidence.e_gain(),
            e_harm = evidence.e_harm(),
            "certification block"
        );
        if evidence.certified_gain(config.alpha) {
            certified = Some(true);
            break;
        }
        if evidence.certified_harm(config.alpha) {
            certified = Some(false);
            break;
        }
    }

    let verdict = CertifyVerdict::snapshot(&evidence, certified, config.alpha);
    tracing::info!(?verdict.certified, cols = verdict.cols, mean_delta = verdict.mean_delta,
        e_gain = verdict.e_gain, "sequential ship certification");
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Deterministic pseudo-noise in ±spread, a pure function of (row, pass).
    fn noise(row: usize, pass: usize, spread: f64) -> f64 {
        let mut hash = 1469598103934665603u64;
        for byte in [row as u64, pass as u64, 0x9e3779b97f4a7c15] {
            hash ^= byte;
            hash = hash.wrapping_mul(1099511628211);
        }
        ((hash % 10_001) as f64 / 10_000.0 - 0.5) * 2.0 * spread
    }

    fn config() -> CertifyConfig {
        CertifyConfig::default()
    }

    #[tokio::test]
    async fn true_effect_certifies_and_stops_early() {
        let verdict = certify_paired(
            400,
            &config(),
            |row, pass| async move { Ok(0.3 + noise(row, pass, 0.25)) },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(verdict.certified, Some(true));
        assert!(verdict.cols < 100, "big effect must stop early, used {}", verdict.cols);
        assert!(verdict.mean_delta > 0.2);
        assert!(verdict.e_gain >= 10.0);
    }

    #[tokio::test]
    async fn harmful_candidate_certifies_harm() {
        let verdict = certify_paired(
            400,
            &config(),
            |row, pass| async move { Ok(-0.3 + noise(row, pass, 0.25)) },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(verdict.certified, Some(false));
        assert!(verdict.e_harm >= 10.0);
    }

    #[tokio::test]
    async fn null_effect_is_refused_with_open_evidence() {
        let verdict = certify_paired(
            300,
            &config(),
            |row, pass| async move { Ok(noise(row, pass, 0.25)) },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(verdict.certified, None);
        assert_eq!(verdict.cols, 300, "a refusal spends every column");
        assert!(verdict.e_gain < 10.0 && verdict.e_harm < 10.0);
    }

    #[tokio::test]
    async fn prior_cannot_fake_a_certification() {
        let mut config = config();
        config.prior = Some(BetPrior { mean: 0.5, sd: 0.05, weight: 100.0 });
        let verdict = certify_paired(
            300,
            &config,
            |row, pass| async move { Ok(noise(row, pass, 0.25)) },
            |_| {},
        )
        .await
        .unwrap();
        assert_ne!(
            verdict.certified,
            Some(true),
            "a wildly optimistic prior shapes bets, never wealth"
        );
    }

    #[tokio::test]
    async fn column_order_is_seeded_and_fixed_up_front() {
        async fn visit_order(seed: u64) -> Vec<usize> {
            let visited = Mutex::new(Vec::new());
            // Zero-effect stream so every column is consumed in order.
            certify_paired(
                40,
                &CertifyConfig { seed, block: 1, ..CertifyConfig::default() },
                |row, _| {
                    visited.lock().unwrap().push(row);
                    async { Ok(0.0) }
                },
                |_| {},
            )
            .await
            .unwrap();
            visited.into_inner().unwrap()
        }
        let first = visit_order(7).await;
        assert_eq!(first, visit_order(7).await, "same seed, same column order");
        assert_ne!(first, visit_order(8).await, "different seed reshuffles");
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..40).collect::<Vec<_>>(), "every row exactly once");
    }

    #[tokio::test]
    async fn passes_average_before_the_wallets_look() {
        let calls = Mutex::new(0usize);
        let verdict = certify_paired(
            50,
            &CertifyConfig { passes: 2, ..CertifyConfig::default() },
            |_, pass| {
                *calls.lock().unwrap() += 1;
                // +0.4 then -0.4 per column: averages to exactly zero.
                async move { Ok(if pass == 0 { 0.4 } else { -0.4 }) }
            },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(*calls.lock().unwrap(), 100, "passes × rows rollout pairs");
        assert_eq!(verdict.certified, None);
        assert_eq!(verdict.mean_delta, 0.0);
        assert_eq!(verdict.e_gain, 1.0);
    }

    #[tokio::test]
    async fn max_cols_caps_the_spend() {
        let verdict = certify_paired(
            300,
            &CertifyConfig { max_cols: Some(30), ..CertifyConfig::default() },
            |row, pass| async move { Ok(noise(row, pass, 0.25)) },
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(verdict.certified, None);
        assert_eq!(verdict.cols, 30);
    }

    #[tokio::test]
    async fn rollout_error_aborts_the_attempt() {
        let result = certify_paired(
            10,
            &config(),
            |row, _| async move {
                if row == 3 { bail!("endpoint down") } else { Ok(0.0) }
            },
            |_| {},
        )
        .await;
        assert!(result.is_err(), "failure policy belongs to the rig, not the pawl");
    }
}
