# PR #103 rendering comparison

The PR already builds one alias-aware MiniJinja input value per formatting call.
This workload measures that implementation and adds regressions for fresh inputs,
alias collisions, skipped fields and recovery after a render error. No duplicate
context-sharing implementation is introduced.

Compare upstream main plus this fixed harness with PR #103 plus the identical
harness. Both use the same lockfile, toolchain, release flags and valid inputs.
The comparison measures the PR's valid-input rendering path as a whole; it does
not isolate the individual effects of error handling and context sharing.

Run `cargo build -p dspy-rs --release --no-default-features --example bench-input-rendering --locked -j1`
on both clean commits. Copy each executable before the next build overwrites it.
Then run `python benchmarks/rendering/run.py --baseline BASE_EXE --baseline-revision BASE_SHA --candidate PR_EXE --out EVIDENCE_DIR`.
Keep evidence outside the checkout. The runner verifies identical committed Rust
harness source, records hashes, and retains every raw sample and output checksum.

Each case contains a 32 KiB plain payload and 1, 8 or 16 ordinary/Jinja string
fields. Aliases, field metadata and the shared payload are accessed by Jinja.
The plan is eight timing samples per executable in ABBA order, 100 warmups and
2,000 iterations per sample. Allocation counts use a separate 200-iteration run;
its timings are not used for latency comparisons. Requested allocation bytes
include reallocations and are not peak live memory or RSS.

Acceptance is predeclared: all six outputs must match byte for byte; default
rendering median latency must remain within 10%; widths 8 and 16 must show at
least 10% lower median Jinja latency and 50% fewer requested allocation bytes.
Width 1 Jinja is descriptive. Raw ranges are not confidence intervals, and a
failed control remains a failed/inconclusive result. Do not trim slow samples.

These local rendering measurements support part of #24/#25; they do not establish
the live inference/I/O speedup requested in #22 or a repository-wide ownership
redesign. Functional error/fallback tests are separate from valid-input timings.
