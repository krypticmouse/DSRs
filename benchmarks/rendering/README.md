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

## Warm system prompts

`bench-system-prompts` measures `Interpreter::conversation_opening` over fixed
signatures containing 1, 8 and 16 input fields. Model construction, load and its
validation are excluded; no HTTP request is sent. The adapter's uncached system
builder is the output oracle. Both executables use byte-identical harnesses and
the same lockfile and release flags.

`run_system.py` checks exact complete prompt bytes before timing eight samples
per executable in ABBA order, with 100 warmups and 20,000 iterations per sample.
Its predeclared gate is at least 10% lower median latency at each width. This
measures repeated unchanged instructions. Scoped-instruction changes, concurrent
bindings, fresh live inputs and graph lifetime have separate functional tests;
cache-miss latency is not measured. The interpreter retains one last-used
instruction/system pair per signature, and never retains live input values.

Run `cargo +1.98.1 build -p dspy-rs --release --no-default-features --example bench-system-prompts --locked -j1`
on each clean ref, then invoke `python benchmarks/rendering/run_system.py
--baseline BASE_EXE --baseline-revision BASE_SHA --candidate PR_EXE --out OUT`.
Retain the executable build provenance separately from the runner manifest.

See [the retained results](RESULTS.md) and the
[ownership decision](../../docs/adr/2026-11-10-runtime-ownership.md).
These rendering measurements do not establish #22's live inference/I/O target.
Functional error/fallback tests are separate from valid-input timings.
