use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::optimizer::gepa::GEPACandidate;

/// Per-example dominance frontier for [`GEPA`](crate::GEPA)'s evolutionary search.
///
/// The key insight: optimizing for average score across examples lets the optimizer
/// overfit to easy examples while ignoring hard ones. The Pareto frontier prevents
/// this by keeping every candidate that's the *best on at least one example*. A
/// candidate that scores 0.3 average but is the only one to crack example #7 stays
/// on the frontier alongside a candidate that scores 0.9 average but fails #7.
///
/// [`GEPA`](crate::GEPA) samples parents from this frontier proportional to coverage
/// (how many examples they win on), so well-rounded candidates get sampled more often
/// but specialists aren't eliminated. Candidates that are dominated on every example
/// get pruned automatically.
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    /// All candidates currently on the frontier
    candidates: Vec<GEPACandidate>,

    /// Maps example index to the candidate IDs that achieve max score on it
    /// example_id -> [candidate_ids]
    example_to_best: HashMap<usize, Vec<usize>>,

    /// Maps candidate ID to the examples it wins on
    /// candidate_id -> [example_ids]
    candidate_to_examples: HashMap<usize, HashSet<usize>>,

    /// Next candidate ID to assign
    next_id: usize,
}

impl ParetoFrontier {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            example_to_best: HashMap::new(),
            candidate_to_examples: HashMap::new(),
            next_id: 0,
        }
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
        // Assign ID to new candidate
        candidate.id = self.next_id;
        self.next_id += 1;

        // Find examples where this candidate achieves max score
        let mut wins_on = HashSet::new();

        for (example_idx, &score) in scores.iter().enumerate() {
            let current_best = self.example_to_best.get(&example_idx).and_then(|best_ids| {
                best_ids
                    .iter()
                    .filter_map(|&id| self.candidates.iter().find(|c| c.id == id))
                    .filter_map(|c| c.example_scores.get(example_idx))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            });

            match current_best {
                Some(best_score) if score > best_score => {
                    // New best for this example
                    wins_on.insert(example_idx);
                }
                Some(best_score) if (score - best_score).abs() < 1e-6 => {
                    // Tied for best
                    wins_on.insert(example_idx);
                }
                None => {
                    // First candidate for this example
                    wins_on.insert(example_idx);
                }
                _ => {}
            }
        }

        // Only add if candidate wins on at least one example
        if wins_on.is_empty() {
            return false;
        }

        // Store scores with candidate
        candidate.example_scores = scores.to_vec();

        // Update mappings
        for &example_idx in &wins_on {
            // Find current max score for this example
            let max_score = scores[example_idx];

            // Remove candidates that are now dominated on this example
            if let Some(best_ids) = self.example_to_best.get_mut(&example_idx) {
                // Keep only candidates with equal or better scores
                best_ids.retain(|&id| {
                    if let Some(existing) = self.candidates.iter().find(|c| c.id == id) {
                        if let Some(&existing_score) = existing.example_scores.get(example_idx) {
                            (existing_score - max_score).abs() < 1e-6 || existing_score > max_score
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                });

                if (max_score - scores[example_idx]).abs() < 1e-6 {
                    best_ids.push(candidate.id);
                }
            } else {
                self.example_to_best.insert(example_idx, vec![candidate.id]);
            }
        }

        self.candidate_to_examples.insert(candidate.id, wins_on);

        // Remove dominated candidates from frontier
        self.prune_dominated();

        // Add new candidate
        self.candidates.push(candidate);

        true
    }

    fn prune_dominated(&mut self) {
        let mut still_winning: HashSet<usize> = HashSet::new();

        for candidate_ids in self.example_to_best.values() {
            still_winning.extend(candidate_ids.iter());
        }

        self.candidates.retain(|c| still_winning.contains(&c.id));
        self.candidate_to_examples
            .retain(|id, _| still_winning.contains(id));
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

        // Calculate coverage for each candidate
        let coverages: Vec<usize> = self
            .candidates
            .iter()
            .map(|c| {
                self.candidate_to_examples
                    .get(&c.id)
                    .map(|examples| examples.len())
                    .unwrap_or(0)
            })
            .collect();

        let total_coverage: usize = coverages.iter().sum();

        if total_coverage == 0 {
            // Fallback to uniform sampling
            return self.candidates.first();
        }

        // Sample proportional to coverage
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
    /// This is what [`GEPA`](crate::GEPA) installs as the final instruction — the
    /// Pareto frontier preserves diversity during search, but the winner is still
    /// picked by average.
    pub fn best_by_average(&self) -> Option<&GEPACandidate> {
        self.candidates.iter().max_by(|a, b| {
            let avg_a = a.average_score();
            let avg_b = b.average_score();
            avg_a.partial_cmp(&avg_b).unwrap()
        })
    }

    pub fn statistics(&self) -> ParetoStatistics {
        let num_candidates = self.candidates.len();
        let num_examples_covered = self.example_to_best.len();

        let coverage_per_candidate: Vec<usize> = self
            .candidates
            .iter()
            .map(|c| {
                self.candidate_to_examples
                    .get(&c.id)
                    .map(|examples| examples.len())
                    .unwrap_or(0)
            })
            .collect();

        let avg_coverage = if !coverage_per_candidate.is_empty() {
            coverage_per_candidate.iter().sum::<usize>() as f32
                / coverage_per_candidate.len() as f32
        } else {
            0.0
        };

        let max_coverage = coverage_per_candidate.iter().copied().max().unwrap_or(0);
        let min_coverage = coverage_per_candidate.iter().copied().min().unwrap_or(0);

        ParetoStatistics {
            num_candidates,
            num_examples_covered,
            avg_coverage,
            max_coverage,
            min_coverage,
        }
    }
}

impl Default for ParetoFrontier {
    fn default() -> Self {
        Self::new()
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
