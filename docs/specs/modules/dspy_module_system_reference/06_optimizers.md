# Optimizers: How They Discover and Modify Modules

## Current Scope Addendum (2026-02-12)

This document is historical DSPy/Python reference material, preserved for context.

It is not the active Rust optimizer/runtime contract for `dspy-rs`. In current V1–V5 typed scope:
- Optimizers compile against typed trainsets (`Vec<Example<S>>`) and typed metrics.
- Internal predictor discovery is Facet-driven and not a public `named_predictors()` surface.
- `_compiled`, `reset_copy()`, and `settings.trace` are not active Rust API contracts.

Refer to the active contracts in:
- `docs/specs/modules/design_reference.md`
- `docs/specs/modules/breadboard.md`

## The Contract

The implicit contract between an optimizer and a module:

1. **The module has `Predict` instances as leaf parameters.** Discovered via `named_parameters()` / `named_predictors()`. A module with no Predict instances has nothing to optimize.
2. **Each Predict has a `signature`** with mutable `.instructions` and field `prefix`/`desc`.
3. **Each Predict has a `demos` list** (initially `[]`). The primary optimization lever.
4. **Each Predict has an optional `lm`** attribute. BootstrapFinetune replaces this with a finetuned model.
5. **Running the module records traces** to `settings.trace`. Optimizers read traces to attribute outputs to specific predictors.
6. **Student and teacher must be structurally equivalent.** Same number of predictors, same names, same signatures.
7. **`deepcopy()` and `reset_copy()` produce valid independent copies.** Optimizers always copy before modifying.
8. **`dump_state()` / `load_state()` round-trip the optimized state.**

---

## 1. Module Discovery

### `named_parameters()` -- What Optimizers See

```python
# For a program like:
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Predict("question -> passages")
        self.answer = dspy.ChainOfThought("question, passages -> answer")

# named_parameters() returns:
[
    ("retrieve", <Predict>),          # self.retrieve IS a Parameter
    ("answer.predict", <Predict>),    # ChainOfThought holds self.predict
]
```

### `named_predictors()` -- Convenience Filter

```python
def named_predictors(self):
    from dspy.predict.predict import Predict
    return [(name, param) for name, param in self.named_parameters()
            if isinstance(param, Predict)]
```

Almost every optimizer uses this. Since `Predict` is currently the only `Parameter` subclass, `named_parameters()` and `named_predictors()` return the same things. But the filter makes the intent explicit.

### `predictor2name` / `name2predictor` Mappings

Optimizers (especially BootstrapFewShot) build bidirectional maps to connect traces back to predictors:

```python
# In BootstrapFewShot._prepare_predictor_mappings():
self.name2predictor = {}
self.predictor2name = {}
for name, predictor in self.student.named_predictors():
    self.name2predictor[name] = predictor
    self.predictor2name[id(predictor)] = name
# Same for teacher
```

`id(predictor)` is the key -- when a trace records `(predictor_instance, inputs, prediction)`, the optimizer looks up `predictor2name[id(predictor_instance)]` to find which named predictor produced that output.

---

## 2. What Optimizers Modify

There are exactly **four** things optimizers touch on Predict instances:

| Property | Type | Modified By | Purpose |
|----------|------|-------------|---------|
| `predictor.demos` | `list[Example]` | BootstrapFewShot, MIPRO, RandomSearch, LabeledFewShot | Few-shot examples prepended to prompt |
| `predictor.signature.instructions` | `str` | COPRO, MIPROv2 | Task instruction text |
| `predictor.signature` field prefixes | `str` | COPRO | Output field prefix text |
| `predictor.lm` | `LM` | BootstrapFinetune, BetterTogether | The language model itself (finetuned) |

Additionally, `program._compiled = True` is set by most optimizers after compilation.

---

## 3. The `compile()` Interface

```python
# dspy/teleprompt/teleprompt.py
class Teleprompter:
    def compile(self, student: Module, *,
                trainset: list[Example],
                teacher: Module | None = None,
                valset: list[Example] | None = None,
                **kwargs) -> Module:
```

**The contract**:
- **Input**: An uncompiled `student` Module and a `trainset` of `Example` objects
- **Output**: A modified copy of the student with optimized parameters
- Most optimizers deep-copy or `reset_copy()` the student first -- never mutating the original
- `student._compiled = True` on the returned module
- Same structure, but with modified demos/instructions/lm on its predictors

---

## 4. Tracing -- How Optimizers Observe Execution

### How Tracing Works

1. `settings.trace` is a global (thread-local) list, initialized via `dspy.context(trace=[])`.

2. Every `Predict._forward_postprocess()` appends to this trace:

```python
def _forward_postprocess(self, completions, signature, **kwargs):
    pred = Prediction.from_completions(completions, signature=signature)
    if settings.trace is not None and settings.max_trace_size > 0:
        trace = settings.trace
        if len(trace) >= settings.max_trace_size:
            trace.pop(0)
        trace.append((self, {**kwargs}, pred))
        # Tuple: (predictor_instance, input_kwargs_dict, prediction_output)
    return pred
```

3. **Optimizers capture traces** by wrapping execution in a trace context:

```python
# BootstrapFewShot:
with dspy.context(trace=[]):
    prediction = teacher(**example.inputs())
    trace = dspy.settings.trace
# trace is now [(pred1, inputs1, output1), (pred2, inputs2, output2), ...]
```

4. **Traces connect predictors to their I/O**: The `predictor_instance` in the tuple lets optimizers map back to named predictors via `predictor2name[id(predictor)]`.

5. **Metrics can use traces**: Metric functions can accept an optional `trace` parameter:
```python
def my_metric(example, prediction, trace=None):
    # Can inspect intermediate steps, not just final output
```

---

## 5. Key Optimizers

### BootstrapFewShot (`dspy/teleprompt/bootstrap.py`)

The foundational optimizer. Populates `demos` on Predict instances by running a teacher and capturing successful traces.

**Step 1: `compile(student, *, teacher, trainset)`**
```python
def compile(self, student, *, teacher=None, trainset):
    self.student = student.reset_copy()  # Deep copy + clear all demos
    self.teacher = (teacher or student).deepcopy()
    self._prepare_predictor_mappings()
    self._bootstrap()
    self._train()
    self.student._compiled = True
    return self.student
```

**Step 2: `_prepare_predictor_mappings()`**
- Asserts student and teacher have identical structure (same number of predictors, same names)
- Builds `name2predictor` and `predictor2name` for both

**Step 3: `_bootstrap()` -- Generate Demo Candidates**

For each training example:
```python
for example in trainset:
    with dspy.context(trace=[]):
        prediction = self.teacher(**example.inputs())
        trace = dspy.settings.trace

    # Check if the output passes the metric
    if self.metric(example, prediction):
        # Extract demos from the trace
        for predictor, inputs, output in trace:
            name = self.predictor2name[id(predictor)]
            demo = dspy.Example(augmented=True, **inputs, **output)
            self.name2traces[name].append(demo)
```

The key mechanism: run the teacher, capture the trace, check the metric, and if it passes, create `Example` objects from each predictor's input/output pair.

**Step 4: `_train()` -- Assign Demos to Student**

For each student predictor:
```python
for name, predictor in self.student.named_predictors():
    augmented_demos = self.name2traces[name][:self.max_bootstrapped_demos]
    raw_demos = self.raw_demos[name][:self.max_labeled_demos]
    predictor.demos = augmented_demos + raw_demos
```

`augmented_demos` are the bootstrapped ones (from successful teacher traces). `raw_demos` are unbootstrapped training examples.

### BootstrapFewShotWithRandomSearch (`dspy/teleprompt/random_search.py`)

Runs BootstrapFewShot multiple times with different configurations and picks the best:

```python
# Generates candidate programs with different strategies:
# Seed -3: Zero-shot (reset_copy, no demos)
# Seed -2: Labels only (LabeledFewShot)
# Seed -1: Unshuffled bootstrap
# Seeds 0+: Shuffled bootstrap with random demo count

# Evaluates each on validation set
# Returns the best-scoring program
# Attaches all candidates as best_program.candidate_programs
```

### MIPROv2 (`dspy/teleprompt/mipro_optimizer_v2.py`)

The most sophisticated optimizer. Jointly optimizes instructions AND demos using Bayesian optimization (Optuna).

**Three-phase process**:

**Phase 1: Bootstrap few-shot examples** (`_bootstrap_fewshot_examples`)
- Uses `create_n_fewshot_demo_sets()` which internally runs multiple BootstrapFewShot compilations
- Produces `demo_candidates[i]` -- a list of demo sets for each predictor `i`

**Phase 2: Propose instruction candidates** (`_propose_instructions`)
- Uses `GroundedProposer` -- an LM-based instruction generator
- Can be program-aware (reads source code), data-aware (summarizes training data), tip-aware (includes prompting tips), fewshot-aware (includes example demos)
- Produces `instruction_candidates[i]` -- a list of instruction strings for each predictor `i`

**Phase 3: Bayesian optimization** (`_optimize_prompt_parameters`)
```python
# Uses Optuna TPE sampler
for trial in optuna_study:
    # For each predictor i:
    instruction_idx = trial.suggest_categorical(f"instruction_{i}", range(n_candidates))
    demos_idx = trial.suggest_categorical(f"demos_{i}", range(n_demo_sets))

    # Apply instruction
    updated_sig = predictor.signature.with_instructions(
        instruction_candidates[i][instruction_idx]
    )
    set_signature(predictor, updated_sig)

    # Apply demos
    predictor.demos = demo_candidates[i][demos_idx]

    # Evaluate the assembled program
    score = evaluate(program, devset=minibatch)
    # Optuna learns which combinations work best
```

### COPRO (`dspy/teleprompt/copro_optimizer.py`)

Pure instruction optimization (no demo manipulation):

```python
for predictor in program.predictors():
    # Generate candidate instructions using an LM
    for breadth iterations:
        candidates = generate_instruction_candidates(current_instruction)

    # Evaluate each candidate
    for candidate in candidates:
        updated_sig = signature.with_instructions(candidate.instruction)
        updated_sig = updated_sig.with_updated_fields(last_key, prefix=candidate.prefix)
        set_signature(predictor, updated_sig)
        score = evaluate(program)

    # Iterate for depth rounds, feeding previous attempts and scores
```

Modifies both `signature.instructions` and the last output field's `prefix`.

### BootstrapFinetune (`dspy/teleprompt/bootstrap_finetune.py`)

Fundamentally different: modifies **model weights** rather than the prompt.

**Step 1: `bootstrap_trace_data()`** -- Run teacher on training set with tracing:
```python
for example in trainset:
    with dspy.context(trace=[]):
        prediction = program(**example.inputs())
        trace = dspy.settings.trace
    score = metric(example, prediction)
    trace_data.append({example, prediction, trace, score})
```

**Step 2: `_prepare_finetune_data()`** -- Convert traces to training format:
```python
for trace_entry in trace_data:
    for pred, inputs, outputs in trace_entry.trace:
        # Use the adapter to format as training data
        training_example = adapter.format_finetune_data(
            signature, demos, inputs, outputs
        )
        # This produces chat-format messages suitable for finetuning
```

**Step 3: `finetune_lms()`** -- Group predictors by LM, finetune:
```python
# If multitask=True: all predictors sharing an LM get one combined finetune job
finetuned_lm = lm.finetune(train_data, ...)
```

**Step 4: Update predictor LMs**:
```python
for predictor in group:
    predictor.lm = finetuned_lm
```

### BetterTogether (`dspy/teleprompt/bettertogether.py`)

Composes prompt optimization and weight optimization in a configurable sequence:

```python
strategy = "p -> w -> p"  # prompt, weight, prompt

# p step: BootstrapFewShotWithRandomSearch
# w step: BootstrapFinetune

for step in strategy:
    if step == "p":
        student = prompt_optimizer.compile(student, trainset=trainset)
    elif step == "w":
        student = weight_optimizer.compile(student, trainset=trainset)
    # Reset _compiled=False for next round, preserve LMs
```

---

## 6. How Evaluate Works

**File**: `dspy/evaluate/evaluate.py`

```python
class Evaluate:
    def __call__(self, program, metric=None, devset=None, ...) -> EvaluationResult:
        def process_item(example):
            prediction = program(**example.inputs())
            score = metric(example, prediction)
            return prediction, score

        results = executor.execute(process_item, devset)
        # results: list of (prediction, score) per example

        ncorrect = sum(score for *_, score in results)
        return EvaluationResult(
            score=100 * ncorrect / ntotal,
            results=results
        )
```

- Uses `ParallelExecutor` for multi-threaded evaluation
- For each example: calls `program(**example.inputs())`, then `metric(example, prediction)`
- `EvaluationResult` (subclass of `Prediction`) has `.score` (percentage) and `.results` (list of `(example, prediction, score)`)
- `failure_score` is used when evaluation fails for an example

---

## 7. The Optimization Surface

Putting it all together, here's what the optimization surface looks like for a typical program:

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Predict("question -> passages")
        self.answer = dspy.ChainOfThought("question, passages -> answer")
```

**Discoverable parameters** (via `named_predictors()`):
1. `"retrieve"` -- Predict with signature `"question -> passages"`
2. `"answer.predict"` -- Predict with signature `"question, passages -> reasoning, answer"`

**Per-predictor optimization knobs**:

| Knob | What | Who Modifies | How |
|------|------|-------------|-----|
| `demos` | Few-shot examples | BootstrapFewShot, MIPRO | `predictor.demos = [Example(...), ...]` |
| `signature.instructions` | Task description | COPRO, MIPRO | `signature.with_instructions("...")` |
| Field `prefix` | Output field label | COPRO | `signature.with_updated_fields(name, prefix="...")` |
| Field `desc` | Field description | (rarely modified) | `signature.with_updated_fields(name, desc="...")` |
| `lm` | The language model | BootstrapFinetune | `predictor.lm = finetuned_lm` |

**What gets saved/loaded**:

When you `program.save("path.json")`, it serializes:
```json
{
    "retrieve": {
        "demos": [...],
        "traces": [],
        "train": [],
        "signature": {
            "instructions": "Given the fields `question`, produce the fields `passages`.",
            "fields": {
                "question": {"prefix": "Question:", "desc": "${question}"},
                "passages": {"prefix": "Passages:", "desc": "${passages}"}
            }
        },
        "lm": null
    },
    "answer.predict": {
        "demos": [...],
        "traces": [],
        "train": [],
        "signature": {
            "instructions": "Optimized instruction here...",
            "fields": {
                "question": {"prefix": "Question:", "desc": "${question}"},
                "passages": {"prefix": "Passages:", "desc": "${passages}"},
                "reasoning": {"prefix": "Reasoning:", "desc": "${reasoning}"},
                "answer": {"prefix": "Answer:", "desc": "${answer}"}
            }
        },
        "lm": null
    }
}
```

The architecture (which modules exist, how they're connected) comes from code. The optimized state (demos, instructions, field metadata) comes from the saved file.
