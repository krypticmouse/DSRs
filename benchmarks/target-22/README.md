# Assessment of issue #22

This is a bounded assessment for the performance fix. It does not change
the issue's 2x target into a narrower CPU-only claim. The user requested all three
baselines: live end-to-end latency, repeated requests with caching and offline
framework CPU overhead.

Both programs predict the capital of France from the same question, task and
field descriptions. Every prediction must equal `Paris`. Both use native
JSON Schema formatting, temperature 0, max_tokens 64, no transport retries and no
tracing subscriber. Both explicitly opt into schema support for the tested custom
endpoint. DSPy's adapter may fall back on an error; the HTTP-count gate rejects
any extra requests. Startup, imports, LM construction and warmups are outside each timing.
Framework-specific prompts and their token counts can differ; this is a complete
typed-prediction workload rather than identical HTTP payloads.

The final experiment opts into root JSON string schemas for the candidate's
single-string output. The pre-scalar baseline retains its object schema, and
DSPy uses JSONAdapter's object schema. These are wire representations of the
same typed result (`answer == "Paris"`); no fields, task instructions or
validation are removed. Startup verifies that the task and both field
descriptions reach the Rust prompt. The baseline compatibility trait is a
documented no-op for the old API; the candidate's inherent methods take
precedence. The exact same Rust workload bytes compile against both revisions.
The optimization is explicitly enabled only on a verified endpoint; numeric,
boolean, multiple and complex outputs and tool agents retain object schemas.

Live mode disables response caching, warms the model once and times one request
per independent sample. It includes real HTTP, model inference, formatting and
parsing. This is warm-model latency, not model-loading latency. Cached mode warms
one response, then makes 1,100 matching calls (100 warmup and 1,000 measured);
the server log must show exactly one HTTP request. DSRs uses foyer's warm memory
tier and DSPy uses its memory cache with disk caching disabled. Backend setup is
excluded. Offline mode uses each project's normal mock provider with 100 warmup
and 1,000 measured calls; the server log must show zero HTTP requests. Mock-provider
implementation differences are included and this does not isolate adapter CPU.
DSPy history is disabled; DSRs global, LM and predictor history capacities are
zero. The cost of each framework's disabling mechanism remains included.
This follow-up contains the integrated native-output, configuration and history
fixes. It is a new frozen experiment, not directly comparable to the earlier
delimiter-only revision. Earlier invalid live/cache runs and valid offline
results remain diagnostic evidence for that earlier protocol only.

Eight samples per framework per mode run in balanced order:
baseline DSRs, candidate DSRs, DSPy, DSPy, candidate DSRs, baseline DSRs, repeated
four times. The baseline and candidate use the same Rust workload source and
Rust 1.98.1 release/cache build flags; the runner verifies the archived baseline
binary and workload hashes. Report before/after DSRs ratios separately from the
DSRs-versus-DSPy target. Preserve all samples,
outliers, stderr, source and binary hashes, model identity and package versions.
Compare medians of per-run mean nanoseconds. Report a 2x success only if all three
speedup ratios reach 2; otherwise leave #22 unresolved. These descriptive ranges
are not confidence intervals. A failure of outputs, request counts or any command
invalidates that run, rather than being treated as an optimization result.

The initial all-mode execution stopped at the first live correctness failure;
its diagnostic artifacts are retained separately. The follow-up controller stops
the remaining samples of a failed mode and continues to the next mode, preserving
the same Rust binary, task, model settings and success criterion. A mode needs all
eight valid samples from both frameworks before it receives a comparison ratio.
An invalid live or cached mode cannot be replaced by a successful offline result.

This single question establishes no general task-quality threshold or production
latency guarantee. It is not the HotpotQA reflection pipeline requested by #23,
and does not establish hardware-independent speedups.

Build the committed clean Rust source with Rust 1.98.1 using
`cargo +1.98.1 build -p dspy-rs --release --no-default-features --features cache --locked --example bench-target-22 -j1`.
Use a Python 3.12 environment with DSPy 3.2.1 and retain all resolved packages.
Point `OLLAMA_BASE_URL` at the same local Ollama/llama3.2 server for both programs.
The runner takes `--rust EXE --candidate-record CANDIDATE_BUILD --baseline BASELINE_EXE --baseline-record MANIFEST
--python PYTHON --base-url URL --ollama-log WSL_PATH
--out OUTSIDE_CHECKOUT`. Run without concurrent builds or model requests.

On Windows, [build.ps1](build.ps1) builds a clean checkout and creates its binary
and hash/command manifest. Use the current checkout for the candidate and a
worktree at `fcf173173da7f2bf29cded39bbb97b35763c1820` for the object-schema baseline.
Serialize both builds with the same target directory. For example, from this
checkout, call `./benchmarks/target-22/build.ps1 -Worktree ../DSRs-perf-baseline
-TargetDir C:/perf-target -OutputDir C:/perf-evidence -Kind baseline`, then repeat
with `-Worktree . -Kind candidate`. Pass the resulting `baseline.exe/json` and
`candidate.exe/json` paths to the runner. Git line-ending settings must preserve
identical Rust workload bytes in both worktrees. Install Rust 1.98.1 and DSPy 3.2.1
before building/running, and use the same Ollama endpoint/model for both programs.
The exact original package versions and build records are in the result archive.

See [RESULTS.md](RESULTS.md) for the completed comparison and its scope.

`--modes cached offline` can collect independent mode groups after a completed
live group. Each mode still needs all eight samples from all three programs.
The controller waits for compiler-idle windows and retains attempts rejected
for Cargo/rustc activity at either sample boundary, retrying the same position
in the fixed order. This exclusion is independent of measured speed; all valid
slow samples remain included. Correctness failures are never retried. Partial
groups cannot supply a ratio. Progress is saved after every attempt, and build
revisions are recorded separately from the controller revision. The default
total deadline for waiting/retrying compiler interference is 30 minutes.
