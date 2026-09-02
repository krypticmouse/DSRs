//! Anytime-valid paired evidence: a betting e-process over paired score
//! deltas (Waudby-Smith & Ramdas, "Estimating means of bounded random
//! variables by betting", JRSS-B 2024).
//!
//! The verification-economics problem this solves (measured on the DocsQA
//! campaign, `docs/handoff-verification-economics.md`): a fixed-size paired
//! race must return a verdict at a preset N, so a true small improvement is
//! "unproven" forever and a pile of them can never be harvested. An
//! e-process has no N: evidence for a candidate accumulates across every
//! shared evaluation it ever participates in, the test may be continued or
//! stopped at ANY data-dependent time, and the type-I guarantee
//! `P(sup_t e_t ≥ 1/α) ≤ α` (Ville's inequality) holds regardless. The fee
//! for certifying an effect self-scales as ~1/Δ²: a +0.15 behavioral fix
//! certifies in a handful of columns, a +0.04 harvested bundle in ~150 —
//! and the fee is paid once per shipped bundle, not once per idea.
//!
//! Two wallets are maintained per comparison:
//!
//! - **gain**: e-process against H0 "challenger ≤ incumbent" — grows when
//!   the challenger is genuinely better; `e_gain ≥ 1/α` certifies the gain.
//! - **harm**: the mirror against H0 "challenger ≥ incumbent" — grows when
//!   the challenger is genuinely worse; used as an honest racing kill rule
//!   (replacing repeated z-peeks, which are anti-conservative: ~25–30%
//!   false kills on true tiny-winners across 4–5 looks at z = 1).
//!
//! Each wallet is a test supermartingale `W_t = Π (1 + λ_i·x_i)` with
//! predictable bets `λ_i ∈ [0, ½]` chosen by the approximate-Kelly (aGRAPA)
//! plug-in `λ = μ̂₊/(σ̂² + μ̂₊²)` computed from the *previous* observations
//! only. Deltas are clamped to `[-scale, scale]` and rescaled to `[-1, 1]`;
//! for 0-to-1 metrics `scale = 1.0`.
//!
//! Caveat for engine use: validity assumes the columns fed to `observe` are
//! not chosen *because of* this pair's deltas. Reserved certification
//! columns (shuffled up front, never touched by the search) satisfy this
//! exactly; race columns (shuffled per generation, shared by every child of
//! the step) satisfy it to first order.

use serde::{Deserialize, Serialize};

/// Hard cap on the betting fraction: keeps every factor `1 + λx ≥ ½ > 0`.
const LAMBDA_MAX: f64 = 0.5;
/// Cap for FIXED mixture components (E5a): a fixed bet needs no drawdown
/// headroom for estimation error, only positivity — `1 + λx ≥ 0.1 > 0`.
const MIX_LAMBDA_MAX: f64 = 0.9;
/// Variance prior (one pseudo-observation at maximum spread of a `[-1, 1]`
/// variable is 1.0; a centered ±½ spread gives 0.25 — the standard WSR
/// choice for scores that live well inside the bounds).
const VAR_PRIOR: f64 = 0.25;
/// Log-wealth cap: prevents `exp` overflow; far above any decision bar.
const LOG_E_CAP: f64 = 500.0;

/// A band of fixed bets mixed with the adaptive aGRAPA bet (E5a power
/// repair, `docs/ratchet-v5-escrow-design.md`). The aGRAPA plug-in stalls
/// on real-but-small effects: with judge noise σ ≈ .2, the running mean
/// that sizes its bet has standard error .03 at 44 columns, so on a
/// +.03-class effect the bet hovers near zero exactly when it should be
/// near Kelly — the measured +.0315 conservative miss. Fixed components
/// sized for the anticipated effect band bet full-size from column 1; a
/// convex mixture of test supermartingales is a test supermartingale, so
/// validity is exact and unchanged. The price is the mixture prior
/// (`−ln weight` of log-wealth for the best component); the payoff is that
/// growth starts immediately instead of after the plug-in warms up.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandBets {
    /// Anticipated paired-delta effect sizes, raw score units. The
    /// campaign-measured real-gain band is .02–.06.
    pub effects: Vec<f64>,
    /// Anticipated between-column spread of the paired delta, raw units
    /// (~.18–.25 on 0-to-1 judge metrics, measured).
    pub sd: f64,
    /// Mixture weight on the adaptive aGRAPA component (clamped to
    /// (0, 1]); the remainder splits uniformly across `effects`. Keeps
    /// large-effect adaptivity while the band carries small effects.
    pub agrapa_weight: f64,
}

impl Default for BandBets {
    /// The measured DocsQA-campaign band: real shipped gains live in
    /// .02–.06 at judge sd ≈ .2 (12 campaigns of empirical prior).
    fn default() -> Self {
        Self { effects: vec![0.02, 0.035, 0.06], sd: 0.2, agrapa_weight: 0.5 }
    }
}

impl BandBets {
    /// A band calibrated to a PRE-COMMITTED column budget: the smallest
    /// effect whose Kelly growth can reach the `1/alpha` bar within
    /// `cols` columns is `Δ_min = σ·sqrt(2·ln(1/α)/cols)`; the band brackets
    /// it. This is the honest sequential form of designing the test for
    /// its budget (Koning & van Meer, arXiv:2501.03982: power at the
    /// design horizon need not be sacrificed for anytime validity).
    pub fn for_horizon(cols: usize, alpha: f64, sd: f64) -> Self {
        let cols = cols.max(1) as f64;
        let alpha = if alpha > 0.0 && alpha < 1.0 { alpha } else { 0.1 };
        let sd = if sd.is_finite() && sd > 0.0 { sd } else { 0.2 };
        let delta_min = sd * (2.0 * (1.0 / alpha).ln() / cols).sqrt();
        Self {
            effects: vec![delta_min, 1.5 * delta_min, 2.5 * delta_min],
            sd,
            agrapa_weight: 0.5,
        }
    }

    /// The fixed Kelly bet for one anticipated effect, scaled units.
    fn lambda(&self, effect: f64, scale: f64) -> Option<f64> {
        if !effect.is_finite() || effect <= 0.0 {
            return None;
        }
        let mu = (effect / scale).clamp(0.0, 1.0);
        let sd = ((self.sd / scale).abs()).clamp(0.0, 1.0);
        let lambda = (mu / (sd * sd + mu * mu)).clamp(0.0, MIX_LAMBDA_MAX);
        (lambda > 0.0).then_some(lambda)
    }
}

/// Per-component wealth state for a [`BandBets`] mixture.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MixtureState {
    /// Fixed betting fractions, scaled units, each in (0, MIX_LAMBDA_MAX].
    lambdas: Vec<f64>,
    /// ln weight of the aGRAPA component and of each fixed component.
    log_w_agrapa: f64,
    log_w_fixed: f64,
    /// Per-fixed-component log wealths, both directions.
    log_gain: Vec<f64>,
    log_harm: Vec<f64>,
}

/// Anytime-valid evidence accumulator over paired deltas
/// `challenger − incumbent`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairedEvidence {
    scale: f64,
    count: usize,
    /// Running sum of scaled deltas.
    sum: f64,
    /// Running sum of squared scaled deltas.
    sum_sq: f64,
    /// Bet-prior pseudo-observations (see [`with_bet_prior`]): enter the
    /// bet plug-in only, never the wealth or the reported mean/count.
    ///
    /// [`with_bet_prior`]: PairedEvidence::with_bet_prior
    #[serde(default)]
    prior_n: f64,
    #[serde(default)]
    prior_sum: f64,
    #[serde(default)]
    prior_sum_sq: f64,
    log_e_gain: f64,
    log_e_harm: f64,
    /// Fixed-bet mixture components (`None` = the v4 aGRAPA-only wallet,
    /// byte-identical behavior).
    #[serde(default)]
    mix: Option<MixtureState>,
}

impl PairedEvidence {
    /// `scale` is the score range: deltas are clamped to `[-scale, scale]`.
    /// Pass `1.0` for 0-to-1 metrics, your ceiling for unnormalized ones.
    /// A non-finite or non-positive scale falls back to `1.0`.
    pub fn new(scale: f64) -> Self {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        Self {
            scale,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            prior_n: 0.0,
            prior_sum: 0.0,
            prior_sum_sq: 0.0,
            log_e_gain: 0.0,
            log_e_harm: 0.0,
            mix: None,
        }
    }

    /// Attaches a [`BandBets`] mixture to a fresh accumulator. Attach
    /// before the first `observe`: mixture weights are fixed at attach
    /// time, and the whole point is betting the band from column 1.
    /// Composes with [`with_bet_prior`](Self::with_bet_prior) (the prior
    /// shapes the aGRAPA component's bets; the band components ignore it).
    /// An empty or degenerate band is a no-op.
    pub fn banded(mut self, band: &BandBets) -> Self {
        let lambdas: Vec<f64> = band
            .effects
            .iter()
            .filter_map(|&effect| band.lambda(effect, self.scale))
            .collect();
        if lambdas.is_empty() {
            return self;
        }
        let w_agrapa = if band.agrapa_weight.is_finite() {
            band.agrapa_weight.clamp(1e-6, 1.0)
        } else {
            0.5
        };
        let w_fixed = (1.0 - w_agrapa).max(0.0) / lambdas.len() as f64;
        if w_fixed <= 0.0 {
            return self;
        }
        let n = lambdas.len();
        self.mix = Some(MixtureState {
            lambdas,
            log_w_agrapa: w_agrapa.ln(),
            log_w_fixed: w_fixed.ln(),
            log_gain: vec![0.0; n],
            log_harm: vec![0.0; n],
        });
        self
    }

    /// Warm-starts the BETS (never the wealth) from prior evidence about
    /// the effect — e.g. the search-time validation delta. Validity is
    /// untouched: a betting e-process may condition its λ on *any* past
    /// data (predictability), while the wealth only ever multiplies over
    /// the certification stream itself. The prior enters the bet plug-in
    /// as `weight` pseudo-observations at `mean` with spread `sd`, so λ
    /// opens near Kelly-optimal for the anticipated effect instead of
    /// ramping from zero — worth ~20–40 columns on a small-effect
    /// certification. A wrong prior costs growth until the real data
    /// outweighs it; it cannot create false certifications.
    pub fn with_bet_prior(scale: f64, mean: f64, sd: f64, weight: f64) -> Self {
        let mut evidence = Self::new(scale);
        if !mean.is_finite() || !sd.is_finite() || !weight.is_finite() || weight <= 0.0 {
            return evidence;
        }
        let mean = (mean / evidence.scale).clamp(-1.0, 1.0);
        let sd = (sd / evidence.scale).clamp(0.0, 1.0);
        evidence.prior_n = weight;
        evidence.prior_sum = weight * mean;
        evidence.prior_sum_sq = weight * (mean * mean + sd * sd);
        evidence
    }

    /// The predictable bet for the wallet whose favorable direction has
    /// past mean `mu` (already sign-adjusted to be the wallet's own view).
    fn bet(mu: f64, var: f64) -> f64 {
        let mu = mu.max(0.0);
        (mu / (var + mu * mu)).clamp(0.0, LAMBDA_MAX)
    }

    /// Feeds one paired delta (raw units, `challenger − incumbent`).
    pub fn observe(&mut self, delta: f64) {
        if !delta.is_finite() {
            return;
        }
        let x = (delta / self.scale).clamp(-1.0, 1.0);
        // Plug-in estimates from PAST observations only (predictability);
        // bet-prior pseudo-observations count as past data here.
        let n = self.count as f64 + self.prior_n;
        let total = self.sum + self.prior_sum;
        let total_sq = self.sum_sq + self.prior_sum_sq;
        let mean = total / (n + 1.0);
        let var = ((total_sq - n * mean * mean).max(0.0) + VAR_PRIOR) / (n + 1.0);
        let lambda_gain = Self::bet(mean, var);
        let lambda_harm = Self::bet(-mean, var);
        self.log_e_gain =
            (self.log_e_gain + (1.0 + lambda_gain * x).ln()).min(LOG_E_CAP);
        self.log_e_harm =
            (self.log_e_harm + (1.0 - lambda_harm * x).ln()).min(LOG_E_CAP);
        if let Some(mix) = &mut self.mix {
            for (j, &lambda) in mix.lambdas.iter().enumerate() {
                mix.log_gain[j] = (mix.log_gain[j] + (1.0 + lambda * x).ln()).min(LOG_E_CAP);
                mix.log_harm[j] = (mix.log_harm[j] + (1.0 - lambda * x).ln()).min(LOG_E_CAP);
            }
        }
        self.count += 1;
        self.sum += x;
        self.sum_sq += x * x;
    }

    /// Feeds every delta in order.
    pub fn observe_all(&mut self, deltas: impl IntoIterator<Item = f64>) {
        for delta in deltas {
            self.observe(delta);
        }
    }

    /// Paired observations consumed so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Mean paired delta in raw score units.
    pub fn mean_delta(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum * self.scale / self.count as f64
        }
    }

    /// Log of the wallet's e-value: the aGRAPA wealth alone (v4 path), or
    /// the log-sum-exp mixture of aGRAPA and fixed-band wealths.
    fn log_e_total(&self, log_agrapa: f64, fixed: impl Fn(&MixtureState) -> &[f64]) -> f64 {
        match &self.mix {
            None => log_agrapa,
            Some(mix) => {
                let mut terms = vec![mix.log_w_agrapa + log_agrapa];
                terms.extend(fixed(mix).iter().map(|&lw| mix.log_w_fixed + lw));
                let max = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
            }
        }
    }

    /// The gain wallet's e-value (≥ `1/α` certifies the challenger better).
    pub fn e_gain(&self) -> f64 {
        self.log_e_total(self.log_e_gain, |m| &m.log_gain).exp()
    }

    /// The harm wallet's e-value (≥ `1/α` certifies the challenger worse).
    pub fn e_harm(&self) -> f64 {
        self.log_e_total(self.log_e_harm, |m| &m.log_harm).exp()
    }

    /// Whether the gain is certified at level `alpha`.
    pub fn certified_gain(&self, alpha: f64) -> bool {
        alpha > 0.0 && self.log_e_total(self.log_e_gain, |m| &m.log_gain) >= (1.0 / alpha).ln()
    }

    /// Whether the harm is certified at level `alpha`.
    pub fn certified_harm(&self, alpha: f64) -> bool {
        alpha > 0.0 && self.log_e_total(self.log_e_harm, |m| &m.log_harm) >= (1.0 / alpha).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Symmetric noisy delta stream around `mu` with sd ≈ `sd` (sum of
    /// three uniforms on [-1, 1] has unit variance: bell-ish, bounded).
    fn stream(rng: &mut StdRng, mu: f64, sd: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let u: f64 = (0..3).map(|_| rng.gen_range(-1.0f64..1.0)).sum();
                mu + sd * u
            })
            .collect()
    }

    #[test]
    fn null_streams_rarely_certify_ever() {
        // Anytime validity: the fraction of null streams that EVER certify
        // (at any of the 400 steps) stays near alpha.
        let mut rng = StdRng::seed_from_u64(7);
        let alpha = 0.05;
        let trials = 400;
        let mut false_gain = 0;
        let mut false_harm = 0;
        for _ in 0..trials {
            let mut ev = PairedEvidence::new(1.0);
            let mut hit_gain = false;
            let mut hit_harm = false;
            for d in stream(&mut rng, 0.0, 0.25, 400) {
                ev.observe(d);
                hit_gain |= ev.certified_gain(alpha);
                hit_harm |= ev.certified_harm(alpha);
            }
            false_gain += hit_gain as usize;
            false_harm += hit_harm as usize;
        }
        // Ville guarantees ≤ 5%; allow simulation slack to 7%.
        assert!(
            (false_gain as f64) < 0.07 * trials as f64,
            "gain false-certification rate too high: {false_gain}/{trials}"
        );
        assert!(
            (false_harm as f64) < 0.07 * trials as f64,
            "harm false-certification rate too high: {false_harm}/{trials}"
        );
    }

    #[test]
    fn large_effects_certify_fast_small_effects_take_longer() {
        let mut rng = StdRng::seed_from_u64(11);
        let alpha = 0.05;
        let time_to_certify = |mu: f64, sd: f64, cap: usize, rng: &mut StdRng| -> Option<usize> {
            let mut ev = PairedEvidence::new(1.0);
            for (i, d) in stream(rng, mu, sd, cap).into_iter().enumerate() {
                ev.observe(d);
                if ev.certified_gain(alpha) {
                    return Some(i + 1);
                }
            }
            None
        };
        // A bcplus-class behavioral fix (+0.15) certifies within a race's
        // worth of columns in the median stream.
        let mut big: Vec<usize> = Vec::new();
        for _ in 0..40 {
            if let Some(t) = time_to_certify(0.15, 0.25, 400, &mut rng) {
                big.push(t);
            }
        }
        assert!(big.len() >= 36, "big effect certified in {}/40 streams", big.len());
        big.sort_unstable();
        assert!(
            big[big.len() / 2] <= 60,
            "median time for +0.15 should be well under 60, got {}",
            big[big.len() / 2]
        );

        // A harvested bundle (+0.05) certifies within a few hundred.
        let mut bundle = 0usize;
        for _ in 0..40 {
            if time_to_certify(0.05, 0.25, 600, &mut rng).is_some() {
                bundle += 1;
            }
        }
        assert!(bundle >= 28, "+0.05 bundle certified in only {bundle}/40 streams by 600");

        // A lone +0.005 candidate essentially never certifies at race scale
        // — the measured DocsQA refusal, now visible as arithmetic. (This is
        // why the certification unit must be the bundle, not the idea.)
        let mut lone = 0usize;
        for _ in 0..40 {
            if time_to_certify(0.005, 0.25, 100, &mut rng).is_some() {
                lone += 1;
            }
        }
        assert!(lone <= 4, "+0.005 at n=100 should stay unproven, certified {lone}/40");
    }

    #[test]
    fn harm_wallet_mirrors_gain_wallet() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut ev = PairedEvidence::new(1.0);
        // A disaster child (−0.3): harm certifies quickly, gain never moves.
        let mut killed_at = None;
        for (i, d) in stream(&mut rng, -0.3, 0.25, 200).into_iter().enumerate() {
            ev.observe(d);
            if killed_at.is_none() && ev.certified_harm(0.1) {
                killed_at = Some(i + 1);
            }
        }
        let killed_at = killed_at.expect("a -0.3 child must certify harm");
        assert!(killed_at <= 40, "disaster kill should be fast, took {killed_at}");
        assert!(!ev.certified_gain(0.5), "gain wallet must not grow on harm");
        assert!(ev.mean_delta() < -0.2);
    }

    #[test]
    fn scale_clamps_and_falls_back() {
        let mut ev = PairedEvidence::new(10.0);
        ev.observe_all([5.0, 5.0, 5.0, 5.0]);
        assert_eq!(ev.count(), 4);
        assert!((ev.mean_delta() - 5.0).abs() < 1e-9);

        // Bad scales fall back to 1.0; non-finite deltas are ignored.
        let mut bad = PairedEvidence::new(f64::INFINITY);
        bad.observe(f64::NAN);
        bad.observe(0.5);
        assert_eq!(bad.count(), 1);
        assert!((bad.mean_delta() - 0.5).abs() < 1e-9);

        let mut neg = PairedEvidence::new(-2.0);
        neg.observe(0.5);
        assert!((neg.mean_delta() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn bet_prior_speeds_certification_but_cannot_fake_it() {
        // Correct prior: certifies faster than the cold start on the same
        // stream.
        let mut rng = StdRng::seed_from_u64(19);
        let alpha = 0.05;
        let mut cold_wins = 0usize;
        let mut warm_faster = 0usize;
        for _ in 0..30 {
            let deltas = stream(&mut rng, 0.06, 0.2, 400);
            let time = |mut ev: PairedEvidence| -> Option<usize> {
                for (i, &d) in deltas.iter().enumerate() {
                    ev.observe(d);
                    if ev.certified_gain(alpha) {
                        return Some(i + 1);
                    }
                }
                None
            };
            let cold = time(PairedEvidence::new(1.0));
            let warm = time(PairedEvidence::with_bet_prior(1.0, 0.06, 0.2, 25.0));
            cold_wins += cold.is_some() as usize;
            match (warm, cold) {
                (Some(w), Some(c)) if w <= c => warm_faster += 1,
                (Some(_), None) => warm_faster += 1,
                _ => {}
            }
        }
        assert!(cold_wins >= 20, "the cold start should usually certify: {cold_wins}/30");
        assert!(
            warm_faster >= 24,
            "the warm start should certify at least as fast almost always: {warm_faster}/30"
        );

        // Wrong optimistic prior on NULL streams: anytime validity must
        // hold regardless — the prior shapes bets, not wealth.
        let mut rng = StdRng::seed_from_u64(23);
        let trials = 300;
        let mut false_certs = 0usize;
        for _ in 0..trials {
            let mut ev = PairedEvidence::with_bet_prior(1.0, 0.2, 0.1, 50.0);
            let mut hit = false;
            for d in stream(&mut rng, 0.0, 0.2, 300) {
                ev.observe(d);
                hit |= ev.certified_gain(alpha);
            }
            false_certs += hit as usize;
        }
        assert!(
            (false_certs as f64) < 0.07 * trials as f64,
            "a wrong bet prior must not inflate false certifications: {false_certs}/{trials}"
        );
    }

    #[test]
    fn wealth_survives_long_winning_streaks_without_overflow() {
        let mut ev = PairedEvidence::new(1.0);
        for _ in 0..100_000 {
            ev.observe(1.0);
        }
        assert!(ev.e_gain().is_finite());
        assert!(ev.certified_gain(0.05));
        assert!(!ev.certified_harm(0.5));
    }

    #[test]
    fn band_bets_stay_anytime_valid_under_null() {
        // The band IS an optimistic bet (it wagers the .02–.06 alternative
        // from column 1), so this is the adversarial validity check: on
        // null streams the mixture must still almost never certify EVER.
        // A convex mixture of test supermartingales is a test
        // supermartingale — Ville bounds the ever-crossing rate by alpha.
        let mut rng = StdRng::seed_from_u64(41);
        let alpha = 0.05;
        let trials = 400;
        let mut false_gain = 0;
        let mut false_harm = 0;
        for _ in 0..trials {
            let mut ev = PairedEvidence::new(1.0).banded(&BandBets::default());
            let mut hit_gain = false;
            let mut hit_harm = false;
            for d in stream(&mut rng, 0.0, 0.2, 400) {
                ev.observe(d);
                hit_gain |= ev.certified_gain(alpha);
                hit_harm |= ev.certified_harm(alpha);
            }
            false_gain += hit_gain as usize;
            false_harm += hit_harm as usize;
        }
        assert!(
            (false_gain as f64) < 0.07 * trials as f64,
            "banded gain false-certification rate too high: {false_gain}/{trials}"
        );
        assert!(
            (false_harm as f64) < 0.07 * trials as f64,
            "banded harm false-certification rate too high: {false_harm}/{trials}"
        );
    }

    #[test]
    fn band_bets_convert_the_conservative_miss_class() {
        // The measured failure this repairs: a +.0315 real effect at judge
        // sd ≈ .2 refused at certification budget (ASSAY, 2026-08-31 —
        // holdout said SIG, the cold aGRAPA wallet stalled). Same streams,
        // same alpha, same 278-column budget (the certify150 spend), only
        // the wallet differs.
        // Measured power at Δ=.0315, sd=.2, α=.1 (200 seeded streams/cap):
        //   cap  90: cold  11/200, banded  35/200   (3.2×)
        //   cap 126: cold  35/200, banded  68/200   (1.9×)
        //   cap 150: cold  43/200, banded  80/200   (1.9×)
        //   cap 278: cold 124/200, banded 141/200
        // The band roughly doubles power at campaign-standard certify
        // budgets (90–150 reserved columns). At 90 columns even the band
        // stays a minority — the ASSAY refusal at that spend was
        // budget-bound too, which is why for_horizon exists.
        let alpha = 0.1;
        let trials = 200;
        for (cap, cold_expect, band_floor) in [(126usize, 35usize, 60usize), (150, 43, 72)] {
            let mut rng = StdRng::seed_from_u64(59);
            let mut cold_certs = 0usize;
            let mut band_certs = 0usize;
            for _ in 0..trials {
                let deltas = stream(&mut rng, 0.0315, 0.2, cap);
                let certifies = |mut ev: PairedEvidence| -> bool {
                    for &d in &deltas {
                        ev.observe(d);
                        if ev.certified_gain(alpha) {
                            return true;
                        }
                    }
                    false
                };
                cold_certs += certifies(PairedEvidence::new(1.0)) as usize;
                band_certs +=
                    certifies(PairedEvidence::new(1.0).banded(&BandBets::default())) as usize;
            }
            assert!(
                cold_certs <= cold_expect + 15,
                "cold power moved unexpectedly at cap {cap}: {cold_certs}/{trials}"
            );
            assert!(
                band_certs >= band_floor,
                "the band must roughly double miss-class power at cap {cap}: cold {cold_certs}, banded {band_certs}"
            );
            assert!(
                band_certs >= cold_certs + 20,
                "banded power must clearly dominate cold at cap {cap}: cold {cold_certs}, banded {band_certs}"
            );
        }
    }

    #[test]
    fn band_bets_keep_big_effects_fast_and_compose_with_prior() {
        // The mixture must not tax the big-win lane: a bcplus-class +0.15
        // still certifies within a race's worth of columns.
        let mut rng = StdRng::seed_from_u64(67);
        let alpha = 0.05;
        let mut times = Vec::new();
        for _ in 0..30 {
            let mut ev = PairedEvidence::with_bet_prior(1.0, 0.15, 0.25, 25.0)
                .banded(&BandBets::default());
            for (i, d) in stream(&mut rng, 0.15, 0.25, 400).into_iter().enumerate() {
                ev.observe(d);
                if ev.certified_gain(alpha) {
                    times.push(i + 1);
                    break;
                }
            }
        }
        assert!(times.len() >= 28, "big effect certified in {}/30 banded streams", times.len());
        times.sort_unstable();
        assert!(
            times[times.len() / 2] <= 60,
            "banded median time for +0.15 should stay well under 60, got {}",
            times[times.len() / 2]
        );
    }

    #[test]
    fn horizon_band_brackets_the_provable_minimum() {
        // At 278 columns and alpha .1, sd .2: the smallest Kelly-provable
        // effect is .2·sqrt(2·ln 10/278) ≈ .0257 — the horizon band must
        // open there and bracket upward.
        let band = BandBets::for_horizon(278, 0.1, 0.2);
        assert!((band.effects[0] - 0.0257).abs() < 0.001, "got {}", band.effects[0]);
        assert!(band.effects[2] > band.effects[1] && band.effects[1] > band.effects[0]);
        // Degenerate inputs fall back instead of panicking.
        let fallback = BandBets::for_horizon(0, -1.0, f64::NAN);
        assert!(fallback.effects.iter().all(|e| e.is_finite() && *e > 0.0));
    }

    #[test]
    fn degenerate_bands_are_no_ops() {
        let empty = PairedEvidence::new(1.0)
            .banded(&BandBets { effects: vec![], sd: 0.2, agrapa_weight: 0.5 });
        let junk = PairedEvidence::new(1.0)
            .banded(&BandBets { effects: vec![-0.5, f64::NAN, 0.0], sd: 0.2, agrapa_weight: 0.5 });
        let all_agrapa = PairedEvidence::new(1.0)
            .banded(&BandBets { effects: vec![0.03], sd: 0.2, agrapa_weight: 1.0 });
        for mut ev in [empty, junk, all_agrapa] {
            ev.observe_all([0.1, -0.1, 0.05]);
            let plain = {
                let mut p = PairedEvidence::new(1.0);
                p.observe_all([0.1, -0.1, 0.05]);
                p
            };
            assert!((ev.e_gain() - plain.e_gain()).abs() < 1e-12);
            assert!((ev.e_harm() - plain.e_harm()).abs() < 1e-12);
        }
    }
}
