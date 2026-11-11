# Inference and I/O performance, 2026-11-10

All three requested comparisons meet the predeclared 2x median-throughput target
against DSPy 3.2.1 on this typed QA workload. Live calls use the explicitly enabled
native string-schema optimization on the verified local Ollama endpoint.

| Mode | Updated DSRs median | DSPy median | DSPy / updated DSRs |
| --- | ---: | ---: | ---: |
| Live, response cache disabled | 567.720 ms | 1,482.966 ms | 2.612x |
| Repeated warm response-cache calls | 36.690 us | 2,191.310 us | 59.725x |
| Offline typed calls with normal mock providers | 23.242 us | 859.644 us | 36.986x |

These are medians of eight independent per-process means per program per mode.
Ranges and every raw observation are retained, including slow observations.
They are descriptive measurements, not confidence intervals or guarantees for
other models, signatures, hardware or cold starts.

## What changed

- Skip history-record construction, prompt serialization and parsed-output copies
  when the global, LM and enclosing caller stores all have zero capacity. Enabled
  caller history still works independently, and trace capture remains independent.
- On a response-cache hit, copy only the response and avoid constructing unused
  provider request messages. Preserve the full-entry inspection API.
- Keep field descriptions and task instructions while avoiding duplicate schema
  text for primitive native outputs. Complex outputs retain their schema text.
- Add `LM::with_scalar_output_schema(true)` for verified endpoints that accept
  root JSON strings. A tool-free signature with one string output can generate
  only the value. The adapter restores its known field name before the common
  alias, coercion and constraint stages; the public typed result is unchanged.

The scalar capability is disabled by default and requires native-schema support.
Custom endpoints also need `with_custom_output_schema(true)`. Numeric/boolean,
multiple and complex outputs and tool agents retain their object schemas. The
experimental numeric scalar path failed live arithmetic checks and was excluded
from production eligibility; the final object fallbacks pass those checks.
Raw history and conversations contain the actual scalar representation when used.

The runnable example supports `--scalar-output`, and `--inspect-only` previews
that format without making a request. See the [LM guide](../../docs/docs/components/lm.mdx).

## Identities and controls

- Object-schema baseline: `fcf173173da7f2bf29cded39bbb97b35763c1820`, on
  `benchmark-scalar-output-baseline`. This is the initial framework optimization
  plus the byte-identical benchmark workload, before the scalar implementation.
- Updated runtime: `03b2005bc23479e186256614c5b6ef645ff31f88`.
- Measurement controller: `e311027` (full revision in the provenance record).
  Later changes to this branch only package the evidence.
- Windows 11, Intel i7-10610U, 4 cores/8 threads, Balanced power plan. Rust 1.98.1,
  LLVM 22.1.8, MSVC, locked release builds, no incremental compilation, and only
  the `cache` Cargo feature. Build commands, compiler identities, workload hashes,
  binary hashes, package versions and host details are retained.
- Ollama 0.40.16 in WSL Ubuntu 24.04, CPU execution, one parallel request slot.
  `llama3.2:latest`, 3.2B Q4_K_M, digest
  `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.
- Same task, field descriptions, question and required typed answer (`Paris`),
  temperature 0, 64 output tokens and zero retries. Both Rust builds use identical
  workload bytes; its documented compatibility trait is a no-op on the baseline
  and the candidate's inherent scalar-capability methods take precedence.
  DSPy uses JSONAdapter with native schema opt-in for this custom endpoint.
- Histories and content logging are disabled in the timed workload. Startup,
  imports, LM/cache construction and warmups are excluded. CPU affinity, clocks
  and thermals are not pinned. Different mock-provider implementations are
  included in the offline comparison; this does not isolate adapter CPU alone.

Each mode uses baseline DSRs, updated DSRs, DSPy, DSPy, updated DSRs, baseline DSRs,
repeated four times. Live runs warm once and time one real request. Cached and
offline runs warm 100 times and measure 1,000 calls. Each process must make exactly
2, 1 or 0 HTTP completions respectively, and every prediction must equal `Paris`.
The live measurement includes warm model inference, transport, rendering and
parsing; the identical prompt is warm in the backend too. Cold model/prompt costs
and whole optimization or dataset-download jobs are outside this experiment.

Live and cached/offline mode groups were collected separately with identical
binaries, workload and controller. The controller waits for idle periods and
rejects attempts with Cargo/rustc activity at either sample boundary, then retries
the same position. Two rejected attempts are retained. Correctness failures are
not retried, and valid slow observations are never excluded. Earlier interrupted
collections and exploratory failures are retained locally and are not pooled
into these completed groups.

Against the pre-scalar DSRs baseline, the updated runtime's live median was 1.759x
faster, cached median was about 0.7% slower, and offline median was 1.157x faster.
The 2x gate is the requested comparison with DSPy, not a claim that every new
change independently doubles the previous DSRs implementation.

## Correctness and evidence

- Final runtime regression suites: 189 passed with default features; 169 passed
  without defaults. Three live tests were ignored in each offline suite.
- Explicit live validation: all nine example workflows and four scalar/fallback
  cases passed, including string, integer, boolean and float outputs.
- Native-format regressions cover aliases, typed demos, independent cache keys,
  field constraints, continuation inputs, config copies, history, JSONL trace
  round trips, strict replay, and object fallbacks for tools/multiple/complex fields.
- The updated Ollama example's offline scalar preview passed with an unreachable
  endpoint. A minimal-build PDB filesystem failure was excluded; after cleaning
  generated package artifacts, the complete retry passed.

See [summary.json](results/2026-11-10-windows/summary.json) for medians, ranges and
all run means, [provenance.json](results/2026-11-10-windows/provenance.json) for
identities and controls, and [raw-samples.zip](results/2026-11-10-windows/raw-samples.zip)
for every accepted/rejected attempt in the final collections, native exits,
HTTP counts, boundary checks, build records and verification logs. The archive's
SHA-256 is in the provenance record. Build and run instructions are in [README.md](README.md).
