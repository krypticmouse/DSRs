//! Offline power diagnostic using HARVEST's actual PairedEvidence wallet.
//! Synthetic IID paired deltas: {-1, 0, +1}, specified mean and SD 0.45.
//! This measures certification power, not benchmark/optimizer performance.
//! cargo run -p dspy-rs --example 41-harvest-power --offline

use dspy_rs::optimizer::evidence::PairedEvidence;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let simulations = 5000usize;
    let sd: f64 = 0.45;
    for mean in [0.0f64, 0.05] {
        let second_moment = sd * sd + mean * mean;
        let p_positive = (second_moment + mean) / 2.0;
        let p_negative = (second_moment - mean) / 2.0;
        for alpha in [0.05f64, 0.1] {
            for horizon in [45usize, 90, 150, 300, 600] {
                let mut rng = StdRng::seed_from_u64(20260905);
                let mut certified = 0usize;
                let mut rejected = 0usize;
                for _ in 0..simulations {
                    // The same search-derived optimistic prior under both
                    // null and alternative; it does not add any wealth.
                    let mut evidence = PairedEvidence::with_bet_prior(1.0, 0.05, sd, 25.0);
                    let mut stopped = false;
                    for col in 1..=horizon {
                        let draw = rng.gen_range(0.0..1.0);
                        let delta = if draw < p_positive {
                            1.0
                        } else if draw < p_positive + p_negative {
                            -1.0
                        } else {
                            0.0
                        };
                        // Consume every draw even after stopping so arm
                        // comparisons use the same synthetic streams.
                        if stopped {
                            continue;
                        }
                        evidence.observe(delta);
                        if col % 8 == 0 || col == horizon {
                            if evidence.certified_gain(alpha) {
                                certified += 1;
                                stopped = true;
                            } else if evidence.certified_harm(alpha) {
                                rejected += 1;
                                stopped = true;
                            }
                        }
                    }
                }
                println!("{}", serde_json::json!({
                    "synthetic": true, "simulations": simulations,
                    "mean": mean, "sd": sd, "alpha_per_challenger": alpha,
                    "paired_columns": horizon,
                    "certification_rate": certified as f64 / simulations as f64,
                    "harm_rate": rejected as f64 / simulations as f64,
                }));
            }
        }
    }
}
