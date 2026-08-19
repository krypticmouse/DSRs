use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::optimizer::engine::{SCORE_EPS, ScoreMatrix};
use crate::optimizer::gepa::GEPACandidate;

/// Per-example dominance frontier for candidate selection.
///
/// The key insight: optimizing for average score across examples lets the optimizer
/// overfit to easy examples while ignoring hard ones. The Pareto frontier prevents
/// this by keeping every candidate that's the *best on at least one example*. A
/// candidate that scores 0.3 average but is the only one to crack example #7 stays
/// on the frontier alongside a candidate that scores 0.9 average but fails #7.
///
/// This is a standalone convenience wrapper over the engine's
/// [`ScoreMatrix`](crate::ScoreMatrix)/[`ParetoView`](crate::ParetoView)
/// bookkeeping — one dominance implementation, two entry points.
/// [`GEPA`](crate::GEPA) itself uses the engine's matrix directly (its scores
/// already live there); use `ParetoFrontier` when you track candidate payloads
/// outside an engine.
///
/// Parents are sampled proportional to coverage (how many examples they win
/// on), so well-rounded candidates get sampled more often but specialists
/// aren't eliminated. Candidates that become dominated on every example are
/// pruned automatically.
#[derive(Debug, Clone, Default)]
pub struct ParetoFrontier {
    /// All recorded scores, including rows of since-pruned candidates (their
    /// columns' maxima are always matched by a surviving candidate, so keeping
    /// them never changes dominance).
    matrix: ScoreMatrix,
    /// Surviving (frontier) candidates, in insertion order.
    candidates: Vec<GEPACandidate>,
    /// Matrix row per surviving candidate, parallel to `candidates`.
    rows: Vec<usize>,
    /// Next candidate ID to assign.
    next_id: usize,
}

impl ParetoFrontier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn candidates(&self) -> &[GEPACandidate] {
        &self.candidates
    }

    /// Adds a candidate if it achieves the best score on at least one example.
    ///
    /// Returns `true` if the candidate made it onto the frontier (won or tied on
    /// at least one example). Candidates already on the frontier that no longer
    /// win on any example are pruned.
    pub fn add_candidate(&mut self, mut candidate: GEPACandidate, scores: &[f32]) -> bool {
        candidate.id = self.next_id;
        self.next_id += 1;
        candidate.example_scores = scores.to_vec();

        // Does it win or tie anywhere against the current frontier?
        let best = self.matrix.pareto();
        let wins_somewhere = scores.iter().enumerate().any(|(example, &score)| {
            match best.best_scores().get(example).copied().flatten() {
                Some(best) => f64::from(score) + SCORE_EPS >= best,
                None => true,
            }
        });
        if !wins_somewhere {
            return false;
        }

        let row = self.matrix.candidates();
        for (example, &score) in scores.iter().enumerate() {
            self.matrix.record(row, example, f64::from(score));
        }
        self.candidates.push(candidate);
        self.rows.push(row);

        // Prune candidates the new arrival dominated everywhere.
        let view = self.matrix.pareto();
        let mut idx = 0;
        while idx < self.candidates.len() {
            if view.wins(self.rows[idx]) == 0 {
                self.candidates.remove(idx);
                self.rows.remove(idx);
            } else {
                idx += 1;
            }
        }

        true
    }

    /// Samples a parent candidate, weighted by how many examples it wins on.
    ///
    /// Well-rounded candidates get sampled more often, but specialists that only
    /// win on one hard example still get a chance. This prevents the search from
    /// collapsing onto a single high-average candidate.
    pub fn sample_proportional_to_coverage(&self) -> Option<&GEPACandidate> {
        if self.candidates.is_empty() {
            return None;
        }

        let view = self.matrix.pareto();
        let coverages: Vec<usize> = self.rows.iter().map(|&row| view.wins(row)).collect();
        let total_coverage: usize = coverages.iter().sum();

        if total_coverage == 0 {
            // Fallback to uniform sampling
            return self.candidates.first();
        }

        let mut rng = rand::thread_rng();
        let mut target = rng.gen_range(0..total_coverage);

        for (candidate, &coverage) in self.candidates.iter().zip(coverages.iter()) {
            if target < coverage {
                return Some(candidate);
            }
            target -= coverage;
        }

        // Fallback (shouldn't happen)
        self.candidates.last()
    }

    /// Returns the candidate with the highest average score across all examples.
    ///
    /// The Pareto frontier preserves diversity during search, but the winner is
    /// still picked by average.
    pub fn best_by_average(&self) -> Option<&GEPACandidate> {
        self.candidates.iter().max_by(|a, b| {
            let avg_a = a.average_score();
            let avg_b = b.average_score();
            avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn statistics(&self) -> ParetoStatistics {
        let view = self.matrix.pareto();
        let coverage_per_candidate: Vec<usize> =
            self.rows.iter().map(|&row| view.wins(row)).collect();

        let avg_coverage = if !coverage_per_candidate.is_empty() {
            coverage_per_candidate.iter().sum::<usize>() as f32
                / coverage_per_candidate.len() as f32
        } else {
            0.0
        };

        ParetoStatistics {
            num_candidates: self.candidates.len(),
            num_examples_covered: view
                .best_scores()
                .iter()
                .filter(|best| best.is_some())
                .count(),
            avg_coverage,
            max_coverage: coverage_per_candidate.iter().copied().max().unwrap_or(0),
            min_coverage: coverage_per_candidate.iter().copied().min().unwrap_or(0),
        }
    }
}

/// Snapshot of the Pareto frontier at a point in the search.
///
/// Useful for plotting convergence. A healthy search has `num_candidates` growing
/// slowly (diversity is maintained) while `avg_coverage` increases (candidates are
/// getting more robust). If `num_candidates` is 1, the search has collapsed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoStatistics {
    /// Candidates currently on the frontier. 1 means the search has converged
    /// (or collapsed) to a single instruction.
    pub num_candidates: usize,
    /// Examples where at least one frontier candidate is the best. Should approach
    /// total eval set size as the search progresses.
    pub num_examples_covered: usize,
    /// Mean examples won per candidate. Higher means candidates are more robust;
    /// lower means more specialization.
    pub avg_coverage: f32,
    /// Most examples won by any single candidate.
    pub max_coverage: usize,
    /// Fewest examples won by any frontier candidate (always >= 1 by construction).
    pub min_coverage: usize,
}
