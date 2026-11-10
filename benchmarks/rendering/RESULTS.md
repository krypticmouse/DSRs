# Rendering measurements, 2026-11-10

Both comparisons used Windows 11, an Intel i7-10610U (4 cores/8 threads), the
Balanced power plan, Rust 1.98.1/LLVM 22.1.8 and locked release builds. The Rust
harness and Cargo.lock were byte-identical between each pair. There was no CPU
affinity or clock/thermal control. Timings ran without concurrent compiler jobs.
Every sample, including slow observations, is retained. Ranges are descriptive,
not confidence intervals, and the results apply to these workloads on this host.

## Alias-aware Jinja input rendering

Baseline `8d7c9cf214990d93bfe639db64834c271639743b` is upstream
`72b247e2bddc7337f923461fb27b4127c7d104f3` plus the fixed harness. Candidate
`0f24d2d29d4fa10a18394408fdf2acc4605b1fe4` is PR #103's bounds/error/context
implementation plus the same harness. This is a comparison of that PR bundle,
not an isolated attribution of each change. Later graph sharing and system
caching are not exercised by this input-only benchmark.

Eight timing samples per executable used ABBA order, 100 warmups and 2,000
iterations. Separate allocation runs used 200 iterations; their timings are not
included in the latency comparison. Requested allocation bytes include reallocs
and are not peak live memory or RSS. All six complete outputs matched exactly.

| Case | Median candidate/baseline latency | Requested-byte ratio |
| --- | ---: | ---: |
| Default, 1 field | 0.988 | 1.000 |
| Default, 8 fields | 0.905 | 1.000 |
| Default, 16 fields | 0.935 | 1.000 |
| Jinja, 1 field (descriptive) | 0.917 | 0.867 |
| Jinja, 8 fields | 0.708 | 0.356 |
| Jinja, 16 fields | 0.674 | 0.269 |

All predeclared gates passed: default rendering stayed within 10%, and Jinja
widths 8/16 used at least 10% less median time and 50% fewer requested bytes.
The wider Jinja cases used 29–33% less time and 64–73% fewer requested bytes.
Raw samples, allocation records, hashes and build provenance are in
[`results/2026-11-10`](results/2026-11-10/manifest.json).

## Warm complete prompt construction

Baseline `46f1c90cf5a655eba81d8a08f0cb4a687daea601` is upstream main plus
the fixed harnesses. Candidate `af0247d010f707ede284be4669270cd321213efd`
adds the bounded system-prompt cache and shared graph ownership to PR #103.
Loading, provider construction and network/model execution were outside timing.
All three complete prompt outputs matched exactly, including an independent
uncached adapter oracle within each executable.

Eight samples per executable used ABBA order, 100 warmups and 20,000 measured
iterations. Candidate/baseline median ratios were 0.291, 0.323 and 0.310 for
1/8/16 fields: about 3.1–3.4x throughput for this repeated-instruction workload.
All three predeclared <=0.90 latency-ratio gates passed. This comparison does not
measure changing-instruction cache misses, allocation counts or live inference.
Raw samples, output hashes and build provenance are in
[`results/2026-11-10-system`](results/2026-11-10-system/manifest.json).

An earlier optimized candidate build ran out of host disk space and produced
no timing result. It was excluded. The accepted candidate above built after
reclaiming generated storage and uses the direct forwarding load future.

The correctness run passed 74 tests covering library behavior, golden prompts,
input rendering, dynamic signatures, Jinja recovery, fresh scoped instructions,
independent model bindings and graph release. That functional run used Rust
1.99.1; the measured executables used Rust 1.98.1. Graph sharing itself has no
separately measured speedup claim.
