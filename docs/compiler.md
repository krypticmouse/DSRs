# Compiling and transforming LM programs

`dspy_rs::ir::compiler::Compiler` optimizes a typed LM call graph into another
loadable `Program`. Its pipeline validates the source, enumerates fusion/split
passes, applies graph rewrites, validates and seals each candidate, measures it
on the fixed validation workload, and promotes only candidates satisfying both
quality and latency gates. The result includes the executable `.dsrs`/JSON
artifact, compiler settings, baseline/final measurements and every proposal's
acceptance/rejection reason. Evaluation and round budgets bound the search.

```rust,ignore
use dspy_rs::ir::compiler::{Compiler, CompilerMeasurement};

// Bake the active overlay first, if instructions/models/demos were optimized.
// program = program.bake(&overlay, Default::default())?;
let report = Compiler::default().compile(&program, |candidate| async move {
    // Load candidate using Interpreter::load_shared and your RuntimeEnv factory.
    // Run the same ordered validation cases with the same model, cache, retry,
    // concurrency and tracing settings. Return observed quality and latency:
    evaluate_candidate(candidate).await // Result<CompilerMeasurement>
}).await?;
std::fs::write("optimized.dsrs", report.program.to_dsrs())?;
```

`CompilerMeasurement` contains per-case quality scores (higher is better),
end-to-end latency in microseconds, observed LM call count and textual feedback.
The default gate allows no quality decrease on any validation case and requires
at least a 1% reduction in mean pipeline latency. Invalid measurements, mismatched
case counts, failed candidate evaluation and quality/performance regressions
cannot replace the incumbent. The evaluator owns representative cases, repeats,
warmups and measurement controls; the compiler does not manufacture latency or
infer production speed from a graph's node count. Its configurable margin is a
selection rule, not a statistical confidence interval.

## Fusion and splitting

`Edit::Fuse { first, second }` merges consecutive independent predictions in one
sequence. Input fields/bindings and model references must match, output names and
aliases must be disjoint, and both leaves must have no demonstrations. It joins
the task instructions and output schema, rewires downstream consumers/exports,
and collects the removed node's parameters. Agents, tools, dependent predictions,
different model selections and demonstrated predictions are excluded.

`Edit::Split { leaf, outputs, name }` moves a nonempty proper subset of a
prediction's output fields into a new independent prediction, preserving input
bindings and model/instruction settings. It partitions schemas/constraints,
allocates distinct parameter slots, and rewires all consumers. It supports
sequence steps and root predictions; splitting a root creates a sequence with
the original exported interface. Demonstrations are excluded. The new leaf name
must be an unused identifier.

Both operations are serializable graph-edit data and run through the same
`Program::edited` validation, garbage collection and hash/index rebuilding as
other edits. The parent remains immutable. Hash-bound overlays must be reminted
or baked first. Compiled artifacts use the normal interpreter and do not require
a separate execution backend.

`Program::legal_edits` exposes `FuseNext`/`SplitOutputs`; `Structural` includes
both in its reflection model's menu and materializes the chosen transformation
from textual execution feedback. Structural retains its quality-improvement
gate. The compiler adds the separate latency gate needed to accept faster
programs with unchanged quality. Split candidates can improve task quality in
Structural even when their extra call does not help latency.

Fusion/splitting can change a model's behavior, so graph validity alone does not
establish semantic equivalence. Validate task quality on held-out examples before
deployment. Functional tests demonstrate identical typed outputs with two
provider requests before fusion, one after fusion, and two after splitting.
Those controlled mock-provider results establish routing/call-count behavior;
they are not a production-provider speedup claim.
