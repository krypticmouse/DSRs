# Augmentation Patterns: How Modules Build on Predict

## The Core Idea

Every DSPy module that does anything interesting is **orchestration on top of Predict**. The module itself is not a parameter -- it's a container. The actual "learning" (demos, instructions) lives entirely inside the Predict instances it holds.

There are exactly **four augmentation patterns** in DSPy:

| Pattern | Mechanism | Modules |
|---------|-----------|---------|
| **Signature Extension** | Modify the signature at `__init__` time, delegate to one Predict | ChainOfThought, MultiChainComparison |
| **Multi-Signature Orchestration** | Multiple Predicts with different signatures, orchestrated in a loop | ReAct, ProgramOfThought |
| **Module Wrapping** | Wrap an arbitrary Module, run it multiple times, select best output | BestOfN, Refine |
| **Aggregation** | Take multiple completions and synthesize/vote | MultiChainComparison, `majority()` |

---

## Pattern 1: Signature Extension

### ChainOfThought -- The Canonical Example

**File**: `dspy/predict/chain_of_thought.py`

```python
class ChainOfThought(Module):
    def __init__(self, signature, rationale_field=None, rationale_field_type=str, **config):
        super().__init__()
        signature = ensure_signature(signature)

        # Default rationale field
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)

        # THE AUGMENTATION: prepend a "reasoning" output field
        extended_signature = signature.prepend(
            name="reasoning",
            field=rationale_field,
            type_=rationale_field_type
        )

        # Single Predict with the extended signature
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
```

**What happens**:
- `"question -> answer"` becomes `"question -> reasoning, answer"`
- The LM is forced to produce `reasoning` *before* `answer`
- `forward()` is a pure passthrough to the single Predict

**What optimizers see**: One Predict at path `"predict"`. They can:
- Add demos to `self.predict.demos`
- Rewrite `self.predict.signature.instructions`
- Rewrite the reasoning field's prefix (e.g., change "Let's think step by step" to something better)

**The Reasoning type trick**: If `rationale_field_type` is the `Reasoning` custom type (instead of `str`), the adapter detects it at call time. If the LM supports native reasoning (o1, o3), the adapter *removes* the reasoning field from the signature and enables the model's built-in chain-of-thought via `reasoning_effort` in lm_kwargs. The LM does its own reasoning internally, and the adapter extracts `reasoning_content` from the response. For non-reasoning models, it falls back to text-based reasoning.

### MultiChainComparison -- Aggregation via Signature Extension

**File**: `dspy/predict/multi_chain_comparison.py`

```python
class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):
        super().__init__()
        self.M = M
        signature = ensure_signature(signature)
        *_, self.last_key = signature.output_fields.keys()  # The final output field name

        # Append M input fields for "student attempts"
        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx+1}",
                InputField(
                    prefix=f"Student Attempt #{idx+1}:",
                    desc="${reasoning attempt}"
                ),
            )

        # Prepend a rationale output field
        signature = signature.prepend(
            "rationale",
            OutputField(
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)
```

**The forward method is unique -- it takes `completions` as input**:

```python
def forward(self, completions, **kwargs):
    attempts = []
    for c in completions:
        rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
        answer = str(c[self.last_key]).strip().split("\n")[0].strip()
        attempts.append(
            f"<<I'm trying to {rationale} I'm not sure but my prediction is {answer}>>"
        )

    kwargs = {
        **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
        **kwargs,
    }
    return self.predict(**kwargs)
```

The pattern: run ChainOfThought M times, feed all M attempts into MultiChainComparison, get a synthesized answer. The signature extension adds the M input slots and a synthesis rationale.

---

## Pattern 2: Multi-Signature Orchestration

### ReAct -- Tool-Using Agent Loop

**File**: `dspy/predict/react.py`

```python
class ReAct(Module):
    def __init__(self, signature, tools, max_iters=20):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        # Convert callables to Tool objects
        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        # Add a "finish" tool that signals completion
        # (returns a dict with the original output field values)
        tools["finish"] = Tool(
            func=lambda **kwargs: "Completed.",
            name="finish",
            desc="Signal task completion.",
            args={name: ... for name in signature.output_fields},
        )
        self.tools = tools
```

**Two separate Predict instances with different signatures**:

```python
        # The action-selection signature
        instr = [
            signature.instructions,
            "You will be given `trajectory` as context.",
            f"Tools: {tool_descriptions}",
            "Finish with the `finish` tool when done.",
        ]
        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        # The extraction signature (uses ChainOfThought)
        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)
```

**The agent loop**:

```python
def forward(self, **input_args):
    trajectory = {}

    for idx in range(self.max_iters):
        # Ask the LM what to do next
        pred = self._call_with_potential_trajectory_truncation(
            self.react, trajectory, **input_args
        )

        # Record the action in trajectory
        trajectory[f"thought_{idx}"] = pred.next_thought
        trajectory[f"tool_name_{idx}"] = pred.next_tool_name
        trajectory[f"tool_args_{idx}"] = pred.next_tool_args

        # Actually execute the tool
        try:
            trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](
                **pred.next_tool_args
            )
        except Exception as err:
            trajectory[f"observation_{idx}"] = f"Execution error: {_fmt_exc(err)}"

        # Break if finish tool was selected
        if pred.next_tool_name == "finish":
            break

    # Extract final answer from the full trajectory
    extract = self._call_with_potential_trajectory_truncation(
        self.extract, trajectory, **input_args
    )
    return dspy.Prediction(trajectory=trajectory, **extract)
```

**Context window handling**: `_call_with_potential_trajectory_truncation` retries up to 3 times on `ContextWindowExceededError`, each time truncating the oldest 4 trajectory entries (one tool call = thought + name + args + observation).

**Parameters exposed to optimizers**: Two Predict instances:
- `self.react` -- the action-selection predictor
- `self.extract.predict` -- the ChainOfThought's internal Predict for extraction

### ProgramOfThought -- Code Generation + Execution

**File**: `dspy/predict/program_of_thought.py`

```python
class ProgramOfThought(Module):
    def __init__(self, signature, max_iters=3, interpreter=None):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.input_fields = signature.input_fields
        self.output_fields = signature.output_fields

        # THREE separate ChainOfThought modules, each with a custom signature:

        # 1. Generate code from inputs
        self.code_generate = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("generate").fields,
                self._generate_instruction("generate")
            ),
        )

        # 2. Regenerate code given previous code + error
        self.code_regenerate = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("regenerate").fields,
                self._generate_instruction("regenerate")
            ),
        )

        # 3. Interpret code output into final answer
        self.generate_output = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("answer").fields,
                self._generate_instruction("answer")
            ),
        )

        self.interpreter = interpreter or PythonInterpreter()
```

**The execution loop**:

```python
def forward(self, **kwargs):
    input_kwargs = {name: kwargs[name] for name in self.input_fields}

    # Step 1: Generate code
    code_data = self.code_generate(**input_kwargs)
    code, error = self._parse_code(code_data)
    if not error:
        output, error = self._execute_code(code)

    # Step 2: Retry on failure
    hop = 1
    while error is not None:
        if hop == self.max_iters:
            raise RuntimeError(f"Max iterations reached: {error}")
        input_kwargs.update({"previous_code": code, "error": error})
        code_data = self.code_regenerate(**input_kwargs)
        code, error = self._parse_code(code_data)
        if not error:
            output, error = self._execute_code(code)
        hop += 1

    # Step 3: Interpret code output
    input_kwargs.update({"final_generated_code": code, "code_output": output})
    return self.generate_output(**input_kwargs)
```

**Signature generation** (`_generate_signature(mode)`):
- `"generate"`: original inputs -> `generated_code: str`
- `"regenerate"`: original inputs + `previous_code: str` + `error: str` -> `generated_code: str`
- `"answer"`: original inputs + `final_generated_code: str` + `code_output: str` -> original outputs

**Parameters exposed to optimizers**: Three ChainOfThought modules, each with an internal Predict:
- `self.code_generate.predict`
- `self.code_regenerate.predict`
- `self.generate_output.predict`

---

## Pattern 3: Module Wrapping

### BestOfN -- Rejection Sampling

**File**: `dspy/predict/best_of_n.py`

```python
class BestOfN(Module):
    def __init__(self, module, N, reward_fn, threshold, fail_count=None):
        self.module = module
        self.N = N
        self.threshold = threshold
        self.fail_count = fail_count or N

        # IMPORTANT: wrapped in lambda to prevent named_parameters() from
        # discovering it (a raw function assigned to self would be walked)
        self.reward_fn = lambda *args: reward_fn(*args)
```

```python
def forward(self, **kwargs):
    best_pred, best_score = None, float("-inf")
    fail_count = 0

    for i in range(self.N):
        with dspy.context(rollout_id=i, temperature=1.0):
            pred = self.module(**kwargs)
        score = self.reward_fn(kwargs, pred)

        if score > best_score:
            best_pred, best_score = pred, score
        if score >= self.threshold:
            return pred  # Good enough, return early
        fail_count += 1
        if fail_count >= self.fail_count:
            break

    return best_pred
```

**Key behaviors**:
- Runs the wrapped module N times at temperature=1.0
- Each run gets a unique `rollout_id` in the context
- Returns the first prediction that meets the threshold, or the best overall
- `self.reward_fn` is wrapped in a lambda specifically to prevent parameter discovery (otherwise `named_parameters()` would try to walk into it)

**Parameters exposed to optimizers**: Whatever `self.module` contains. BestOfN itself adds no Predict instances.

### Refine -- BestOfN With Feedback

**File**: `dspy/predict/refine.py`

Refine does everything BestOfN does, plus: after a failed attempt, it generates per-module advice and injects it as a "hint" on retry.

**The feedback mechanism**: Uses `dspy.Predict(OfferFeedback)` to generate advice:

```python
# OfferFeedback signature:
# input_data, output_data, metric_value, output_field_name -> feedback
feedback_pred = dspy.Predict(OfferFeedback)
```

**The hint injection** uses a `WrapperAdapter`:

```python
class WrapperAdapter(adapter.__class__):
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        # Dynamically add a hint field to the signature
        inputs["hint_"] = advice.get(signature2name[signature], "N/A")
        signature = signature.append(
            "hint_",
            InputField(desc="A hint to the module from an earlier run")
        )
        return adapter(lm, lm_kwargs, signature, demos, inputs)
```

**This is the modern replacement for Assert/Suggest**. Instead of backtracking and mutating signatures permanently, Refine:
1. Runs the module
2. If the metric fails, asks an LM for advice
3. Injects that advice as a temporary "hint" field on the next attempt
4. The signature modification happens at call time via the adapter wrapper, not at construction time

---

## Pattern 4: Aggregation

### `majority()` -- Voting

Not a module, just a function:

```python
def majority(prediction_or_completions, normalize=...):
    """Returns the most common value across completions."""
```

### MultiChainComparison (covered above)

Takes M completions and synthesizes them. This is aggregation *via* signature extension.

---

## Deprecated / Removed Modules

### Retry -- Removed

The entire file (`dspy/predict/retry.py`) is commented out. Not exported. Replaced by `Refine` and `BestOfN`.

### Assert / Suggest -- Removed in DSPy 2.6

These were inline constraints that triggered backtracking:
```python
# OLD (removed):
dspy.Assert(len(answer) < 100, "Answer too long")
```

When the constraint failed, it would dynamically modify the signature by adding `past_{output_field}` InputFields and a `feedback` InputField. On persistent failure, `Assert` raised an error; `Suggest` logged and continued.

Replaced by `Refine` which does the same thing more cleanly.

### ChainOfThoughtWithHint -- Removed

Absorbed into `Refine`'s hint injection mechanism.

---

## Summary: What Each Module Exposes to Optimizers

| Module | # Predicts | Paths | What's Optimizable |
|--------|-----------|-------|-------------------|
| **Predict** | 1 | `self` | demos, signature.instructions, field prefixes |
| **ChainOfThought** | 1 | `predict` | demos, instructions, reasoning prefix |
| **MultiChainComparison** | 1 | `predict` | demos, instructions, rationale prefix |
| **ReAct** | 2 | `react`, `extract.predict` | demos and instructions for both action selection and extraction |
| **ProgramOfThought** | 3 | `code_generate.predict`, `code_regenerate.predict`, `generate_output.predict` | demos and instructions for code gen, code regen, and output interpretation |
| **BestOfN** | varies | whatever `self.module` contains | pass-through to wrapped module |
| **Refine** | varies + 1 | wrapped module + feedback predictor | pass-through + feedback generation |

**The invariant**: Every optimizable thing is a Predict. Every Predict has a signature and demos. Modules are just orchestration.
