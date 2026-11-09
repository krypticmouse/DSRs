# COPRO candidate batching: controlled local run

Baseline: `d8659b75b4f9ff8dfde6edc9e6198140430d1a0a` on `feature-grounded-prompt-optimization`. Candidate: the opt-in `candidate_batch_size` implementation on that branch. The same candidate binary ran batch size 1 (existing sequential behavior) and 5; this isolates the scheduling choice, not a full before/after build comparison.

Contract: identical candidate proposals, rollout count, selected instruction, and capped-budget stopping. Hypothesis: sharing the engine's eight rollout slots among five candidates reduces latency when each candidate has too few examples to fill them. Before confirmation, acceptance required a median reduction of at least 30% for one example and no more than 15% median regression for eight examples.

Harness: [`crates/dspy-rs/tests/bench_copro_batching.rs`](../crates/dspy-rs/tests/bench_copro_batching.rs), SHA-256 `AEBC2DA2DB1FB95A0ADA2041E3C954694E6AAA88FCC55D19C15DFA01CE6184D3`; test executable SHA-256 `1E252320BBF390B1D06512A25050029A25A23BDA821B3D1C16C58DCD6F8D5774`. Each rollout awaits 25 ms; the mock proposal model returns four fixed instructions. Breadth 5, depth 1, concurrency 8, one warm-up per configuration per process, then order `1,5,5,1,1,5,5,1`. Two serialized confirmation processes supplied eight samples per configuration. Timing covers the entire `compile_module` call. Native exits were 0.

Environment: Windows 11 Pro 10.0.26100, Intel Core i7-10610U, 8 logical processors; Rust 1.98.1; locked dependencies; `--no-default-features`; test profile with debug level 1 and incremental disabled; Cargo build jobs 2. This was a normal desktop session, with no fixed CPU clock or isolated system load.

| Examples | Batch | Median ms | Range ms | Rollouts | Median ratio to batch 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 282.304 | 231.829–366.676 | 5 | 1.000 |
| 1 | 5 | 114.757 | 86.557–164.550 | 5 | 0.407 |
| 8 | 1 | 292.336 | 252.796–517.267 | 40 | 1.000 |
| 8 | 5 | 276.215 | 234.912–547.875 | 40 | 0.945 |

Raw elapsed milliseconds, in execution order within each configuration (the harness retains the interleaved sample order):

| Process | Examples | Batch 1 | Batch 5 |
| ---: | ---: | --- | --- |
| 1 | 1 | 260.9385, 297.9031, 231.8293, 271.5756 | 110.9410, 118.5732, 97.9764, 86.5573 |
| 2 | 1 | 366.6762, 244.7005, 293.0314, 337.8141 | 155.1238, 96.3179, 164.5502, 132.9953 |
| 1 | 8 | 517.2670, 298.1312, 273.9925, 282.7259 | 547.8749, 368.5358, 254.5528, 248.1413 |
| 2 | 8 | 252.7959, 496.1579, 292.1467, 292.5262 | 249.3548, 234.9124, 299.5845, 297.8763 |

All benchmark runs asserted the same rollout count and selected instruction. Focused COPRO tests also covered capped-budget stopping; the benchmark correctness test and grounded-optimization integration suite passed. The changed Rust files pass `rustfmt --check`. Package-wide `cargo fmt --check` reports existing formatting differences in unrelated examples and tests. The one-example workload met the preset threshold; the eight-example workload showed no median regression. The desktop timing spread is substantial, and this synthetic delay does not establish a production-provider speedup. The option defaults to 1 so existing request ordering stays sequential.
