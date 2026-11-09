# MIPRO trial batching: controlled local run

Baseline: `bbae7796ad299ebe5645cdef561d6a568d6590fc` on `feature-grounded-prompt-optimization`. Candidate: opt-in `candidate_batch_size` on that branch. The same candidate binary ran the default batch size 1 and requested batch size 5; this isolates the scheduling choice, not a separate-build before/after comparison.

Contract: seeded trial combinations, rollout count, selected instruction pair, interval winner selection, and capped-budget stopping remain the same. Hypothesis: sharing eight rollout slots among trials reduces latency when one trial has too few examples to fill them. Before confirmation, acceptance required at least 30% lower one-example median and no more than 15% higher eight-example median.

Harness: [`crates/dspy-rs/tests/bench_mipro_batching.rs`](../crates/dspy-rs/tests/bench_mipro_batching.rs), SHA-256 `C4CB81BBFE8765C1FCEEB25BCB46C5005F0126019A737A45A244AD951D6C335D`; test executable SHA-256 `5EC8F8D4368EBA729FBEDB1D00EA5B4B11525252A8ABD78F1E0DC4EFA3C87F1B`. Two predictors, five instruction choices per predictor, 20 seeded joint trials, full validation every five trials, eight rollout slots, and a 25 ms sleep per rollout. A deterministic metric derived from the candidate hash makes winner equality meaningful. Each process warmed both modes, then ran `1,5,5,1,1,5,5,1`; two serialized processes supplied eight samples per configuration. Timing covers `compile_module`. Native exits were 0.

Environment: Windows 11 Pro 10.0.26100, Intel Core i7-10610U, 8 logical processors; Rust 1.98.1; locked dependencies; `--no-default-features`; test profile with debug level 1 and incremental disabled; Cargo build jobs 2. Desktop load and CPU clocks were not controlled.

| Examples | Requested batch | Effective batch | Median ms | Range ms | Rollouts | Median ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 523.994 | 490.474–566.662 | 16 | 1.000 |
| 1 | 5 | 5 | 169.260 | 152.796–178.744 | 16 | 0.323 |
| 8 | 1 | 1 | 500.743 | 463.628–576.389 | 120 | 1.000 |
| 8 | 5 | 1 | 500.211 | 486.929–512.420 | 120 | 0.999 |

Raw elapsed milliseconds, in execution order within each configuration (interleaved by the harness):

| Process | Examples | Batch 1 | Requested batch 5 |
| ---: | ---: | --- | --- |
| 1 | 1 | 490.4744, 521.2019, 566.6616, 490.5830 | 169.2842, 173.9341, 152.7957, 178.7441 |
| 2 | 1 | 535.1088, 526.7862, 535.8630, 519.7753 | 168.9002, 156.8588, 170.9071, 169.2348 |
| 1 | 8 | 503.3787, 498.1078, 505.7243, 487.7897 | 486.9290, 489.4266, 498.9852, 487.7329 |
| 2 | 8 | 508.4434, 485.1752, 463.6276, 576.3888 | 508.3179, 504.7268, 501.4365, 512.4199 |

All confirmation runs asserted the same rollout count and selected instructions. The implementation caps effective batch size at `max(1, floor(concurrency / minibatch_size))`, so the eight-example case takes the sequential path: batching has no spare rollout slots to use there. A preliminary uncapped prototype exceeded the eight-example regression threshold under a different constant-score harness; those exploratory samples are not comparable to the confirmation above. The result meets the preset threshold for this synthetic workload. It does not establish a production-provider speedup or model-quality equivalence under concurrent provider requests.
