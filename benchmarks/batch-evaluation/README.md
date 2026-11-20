# Trainset evaluation and optimizer batch planning

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
revision above. Build in each checkout and save its executable before the next
build.

```text
cargo +1.100.0 build --locked --release --no-default-features -p dspy-rs --example bench-batch-evaluation
python run.py --baseline path/to/baseline.exe --candidate path/to/candidate.exe --output path/to/new-results
```

The [raw CSV](results/2026-11-20/samples.csv) includes warm-ups, timing in
microseconds, execution order, and native exits. The
[summary](results/2026-11-20/summary.json) records executable hashes, observed
ranges, medians, and paired ratios.
