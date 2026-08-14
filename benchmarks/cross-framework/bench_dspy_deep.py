"""DSPy counterpart of DSRs' 99-deep-harness-bench: deep/heavy pipeline shapes.

Same shapes as the Rust bench where DSPy supports them: sequential chains,
fan-out (executed sequentially — DSPy modules don't parallelize sub-calls unless
you thread manually, which is itself a finding), demo-heavy prompts, wide output
signatures, and context-heavy prompts. Tool loops are omitted: faithfully mocking
DSPy's ReAct protocol with DummyLM is fragile enough that numbers wouldn't be
trustworthy.
"""

import time

import dspy

try:
    from dspy.utils.dummies import DummyLM
except ImportError:
    from dspy.utils import DummyLM


class Step(dspy.Signature):
    """Transform the text one step."""

    text: str = dspy.InputField()
    result: str = dspy.OutputField()


STEP_ANSWER = {"result": "Processed step output with a plausible amount of text in it."}
WIDE_FIELDS = [f"f{i:02d}" for i in range(1, 17)]
Wide16 = dspy.Signature(
    "text -> " + ", ".join(WIDE_FIELDS),
    "Produce all sixteen analysis fields.",
)
WIDE_ANSWER = {name: f"value for field {name}" for name in WIDE_FIELDS}


def make_lm(entries):
    try:
        return DummyLM(entries, cache=False)
    except TypeError:
        return DummyLM(entries)


class Chain(dspy.Module):
    def __init__(self, depth):
        super().__init__()
        self.stages = [dspy.Predict(Step) for _ in range(depth)]

    def forward(self, text):
        for stage in self.stages:
            text = stage(text=text).result
        return dspy.Prediction(result=text)


class FanOut(dspy.Module):
    """16 branches + aggregate. NOTE: branches run sequentially — that is how a
    plain DSPy module executes; concurrency requires manual threading."""

    def __init__(self, width=16):
        super().__init__()
        self.branches = [dspy.Predict(Step) for _ in range(width)]
        self.aggregate = dspy.Predict(Step)

    def forward(self, text):
        outputs = [branch(text=text).result for branch in self.branches]
        return dspy.Prediction(result=self.aggregate(text=" | ".join(outputs)).result)


def timed(label, iters, calls_per_op, fn, warmup=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    per_op = (time.perf_counter() - t0) / iters
    print(
        f"{label:<46} {per_op * 1e6:>10.1f} us/op {per_op * 1e6 / calls_per_op:>8.1f} us/LM-call"
    )


if __name__ == "__main__":
    print(f"dspy {dspy.__version__}")
    print(f"{'shape':<46} {'per run':>13} {'per LM call':>15}")

    # 1-2. Sequential chains.
    for depth, iters in [(10, 300), (50, 60)]:
        lm = make_lm([dict(STEP_ANSWER) for _ in range((iters + 20) * depth)])
        dspy.settings.configure(lm=lm)
        chain = Chain(depth)
        timed(f"chain depth {depth}", iters, depth, lambda: chain(text="start"))

    # 3. Fan-out 16 + aggregate (sequential in DSPy).
    iters = 150
    lm = make_lm([dict(STEP_ANSWER) for _ in range((iters + 20) * 17)])
    dspy.settings.configure(lm=lm)
    fanout = FanOut(16)
    timed("fan-out 16 + aggregate (sequential)", iters, 17, lambda: fanout(text="start"))

    # 4-5. Demo-heavy prompts.
    for demo_count, iters in [(16, 500), (64, 200)]:
        lm = make_lm([dict(STEP_ANSWER) for _ in range(iters + 20)])
        dspy.settings.configure(lm=lm)
        predict = dspy.Predict(Step)
        predict.demos = [
            dspy.Example(
                text=f"Demo input {i} with a realistic sentence of content.",
                result=f"Demo output {i} with a realistic sentence of content.",
            ).with_inputs("text")
            for i in range(demo_count)
        ]
        timed(f"forward with {demo_count} demos", iters, 1, lambda: predict(text="run"))

    # 6. Wide signature (16 output fields).
    iters = 500
    lm = make_lm([dict(WIDE_ANSWER) for _ in range(iters + 20)])
    dspy.settings.configure(lm=lm)
    wide = dspy.Predict(Wide16)
    timed(
        "forward with 16 output fields",
        iters,
        1,
        lambda: wide(text="Analyze this input across all dimensions."),
    )

    # 7. Context-heavy prompt (~50KB input).
    iters = 300
    big_context = "The quick brown fox jumps over the lazy dog. " * 1150
    lm = make_lm([dict(STEP_ANSWER) for _ in range(iters + 20)])
    dspy.settings.configure(lm=lm)
    predict = dspy.Predict(Step)
    timed("forward with 50KB context", iters, 1, lambda: predict(text=big_context))
