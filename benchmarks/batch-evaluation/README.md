# Trainset evaluation and optimizer batch planning

## Original Rust 1.100.0 comparison

The baseline is upstream main at `72b247e2bddc7337f923461fb27b4127c7d104f3`.
The candidate runtime is `56b5fd075b28fd60ca4fe84929dec50273f169c4`.
Each checkout adds the same benchmark example for the build. The harness
SHA-256 is `ae88520bd24f035f1c4a92f027cd146d3b79592d2e489a8f64f5169bb73d6ece`.
Both builds use the same Cargo.lock, SHA-256
`afe48fc92e98a05bfe981f5cd826f39ef73a5b741c979bfb79d28af95ea6e36f`.

The host runs Windows 11 Pro 10.0.26100 with an Intel Core i7-10610U
(4 cores, 8 logical processors) on the Balanced power plan. The release binaries
use Rust and Cargo 1.100.0 for x86_64-pc-windows-msvc, LLVM 23.1.5, locked
dependencies, and default features disabled. Python 3.14.6 runs the controller.
The system uses its normal clock, affinity, and thermal settings.

## Workloads

- **Uneven evaluator latency.** Evaluate 64 synthetic rows with eight slots. Every
  eighth row waits 40 ms. The other rows wait 2 ms. The metric checks the output
  and trace for each row, and the harness checks returned score order.
- **Equal-delay evaluator control.** Evaluate the same rows with eight slots
  and a 5 ms requested delay for every row.
- **Optimizer budget preflight.** Submit eight distinct candidates against 512
  or 4,096 examples. A zero metric-call budget exercises the public batch
  planning path. The harness checks the required rollout count and zero spend.
  Candidate registration and target construction are outside the timed region.

Each workload starts with one warm-up process per binary. Four ABBA blocks
(baseline, candidate, candidate, baseline) then yield eight timed processes
per binary. The controller runs them serially. A paired candidate/baseline
latency ratio of at most 0.80 is the improvement threshold for uneven evaluation
and both preflight workloads. The equal-delay control uses a ceiling of 1.10.

## Results

| Workload | Baseline median [range], ms | Candidate median [range], ms | Median paired ratio |
| --- | ---: | ---: | ---: |
| Uneven evaluator latency | 393.71 [375.64–396.80] | 169.38 [145.67–178.37] | 0.431 |
| Equal-delay evaluator control | 112.34 [84.48–130.80] | 128.83 [102.76–129.49] | 1.069 |
| Preflight, 4,096 pairs | 5.14 [4.17–6.62] | 1.90 [1.71–2.57] | 0.382 |
| Preflight, 32,768 pairs | 251.42 [237.91–258.26] | 17.09 [15.81–19.70] | 0.068 |

All four paired-ratio thresholds passed. The equal-delay control's median
paired increase was 6.9%. The separate medians were 112.34 and 128.83 ms
with overlapping observed ranges. These are descriptive timings for the
local echo workload and the optimizer's budget preflight.

Both builds and all benchmark invocations exited successfully. The host had
no Cargo, rustc, or linker process at either timing boundary. The candidate
also passed 20 focused tests on each of Rust 1.97.1, 1.98.1, and 1.99.1,
covering evaluation order, errors, cached and repeated pairs, and both optimizer
execution paths. The workspace check with all targets and features passed on
Rust 1.97.1.

## Reproduction

Copy [harness.rs](harness.rs) to
`crates/dspy-rs/examples/bench-batch-evaluation.rs` in a checkout of each
revision above. Use distinct Cargo target directories for the two checkouts
so each build compiles its own runtime source.

```text
cargo +1.100.0 build --locked --release --no-default-features -p dspy-rs --example bench-batch-evaluation
python run.py --baseline path/to/baseline.exe --candidate path/to/candidate.exe --output path/to/new-results
```

The [raw CSV](results/2026-11-20/samples.csv) includes warm-ups, timing in
microseconds, execution order, and native exits. The
[summary](results/2026-11-20/summary.json) records executable hashes, observed
ranges, medians, and paired ratios.

## Rust 1.94.0 comparison

Both release builds use Rust and Cargo 1.94.0 for x86_64-pc-windows-msvc,
LLVM 21.1.8, and the same Windows host and Balanced power plan described above.
The workload, harness and ABBA schedule are the same as in the original run.

The candidate source is `84d1caad20dab2464ee5567283d638a64efb6f50`. For the
baseline, start at upstream `72b247e2bddc7337f923461fb27b4127c7d104f3` and copy
`Cargo.toml`, `Cargo.lock`, and the five `crates/*/Cargo.toml` files from that
candidate. This produced the local baseline source
`db104f6679eb47537d3e64acdd0f9ff860415cb9`. Both binaries used Cargo.lock
SHA-256 `6200d330c3cbac5d41f50cd7499d6b4bb603aaa80d59c5f72bd463b43f4799ac`.
The published PR and the benchmark candidate resolve the same normal and build
dependency tree for the benchmark feature set.

Use the reproduction steps above with `cargo +1.94.0`. The initial candidate
build reused the baseline executable through a shared Cargo target. It was
rejected before timing. Clearing the generated `dspy-rs` release artifacts in
the benchmark target forced a rebuild and produced a distinct executable.
Both compiler-idle checks passed and all 72 recorded observations have a
successful native exit.

| Workload | Baseline median [range], ms | Candidate median [range], ms | Median paired ratio |
| --- | ---: | ---: | ---: |
| Uneven evaluator latency | 389.64 [380.24–462.16] | 165.02 [140.67–185.75] | 0.399 |
| Equal-delay evaluator control | 110.62 [88.76–122.57] | 109.36 [90.84–124.51] | 1.005 |
| Preflight, 4,096 pairs | 11.30 [9.95–17.51] | 3.83 [2.99–6.51] | 0.339 |
| Preflight, 32,768 pairs | 549.04 [528.34–564.54] | 28.99 [25.64–32.97] | 0.052 |

All four predefined paired-ratio limits passed. The equal-delay paired ratio
was 1.005 against a 1.10 limit.

- [Raw samples, including warm-ups and native exits](results/rust-1.94.0/samples.csv)
- [Summary and executable hashes](results/rust-1.94.0/summary.json)
