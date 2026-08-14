"""DSPy counterpart of DSRs' 97-perf-microbench: framework overhead with a mocked LM.

Phases mirror the Rust bench: same signature shape (question+context -> answer+confidence),
0-demo and 2-demo forward, plus an orchestration test (256 examples, 20ms simulated
provider latency, concurrency 16).
"""

import time

t_import = time.perf_counter()
import dspy  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - t_import

try:
    from dspy.utils.dummies import DummyLM
except ImportError:
    from dspy.utils import DummyLM


class BenchQA(dspy.Signature):
    """Answer the question using the context. Be concise and accurate."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: float = dspy.OutputField()


CONTEXT = (
    "France is a country in Western Europe. Its capital and largest city is Paris, "
    "known for the Eiffel Tower and the Louvre."
)
ANSWER = {"answer": "Paris is the capital of France.", "confidence": "0.95"}


def make_lm(n):
    try:
        return DummyLM([dict(ANSWER) for _ in range(n)], cache=False)
    except TypeError:
        return DummyLM([dict(ANSWER) for _ in range(n)])


def bench_forward(demos, iters, warmup=100):
    lm = make_lm(iters + warmup)
    dspy.settings.configure(lm=lm)
    predict = dspy.Predict(BenchQA)
    if demos:
        predict.demos = [
            dspy.Example(
                question=f"Demo question {i}?",
                context=f"Demo context {i} with enough text to look like a real retrieval chunk for the benchmark.",
                answer=f"Demo answer {i}.",
                confidence=0.9,
            ).with_inputs("question", "context")
            for i in range(demos)
        ]

    for i in range(warmup):
        predict(question=f"warm {i}", context=CONTEXT)

    t0 = time.perf_counter()
    for i in range(iters):
        # Vary the question so no cache layer can short-circuit the call.
        predict(question=f"What is the capital of France? {i}", context=CONTEXT)
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def bench_orchestration(n_examples=256, threads=16, sleep_s=0.02):
    class SleepEcho(dspy.Module):
        def forward(self, question, **kwargs):
            time.sleep(sleep_s)
            return dspy.Prediction(answer=question)

    devset = [
        dspy.Example(question=str(i), answer=str(i)).with_inputs("question")
        for i in range(n_examples)
    ]

    def exact(example, pred, trace=None):
        return example.answer == pred.answer

    evaluator = dspy.Evaluate(
        devset=devset,
        metric=exact,
        num_threads=threads,
        display_progress=False,
        display_table=False,
    )
    program = SleepEcho()
    t0 = time.perf_counter()
    result = evaluator(program)
    wall = time.perf_counter() - t0
    ideal = (n_examples / threads) * sleep_s
    score = getattr(result, "score", result)
    return wall, ideal, score


if __name__ == "__main__":
    print(f"dspy {dspy.__version__}  (import: {IMPORT_SECONDS:.2f}s)")

    per_call_0 = bench_forward(demos=0, iters=2000)
    print(f"forward end-to-end (0 demos, dummy LM)   {per_call_0 * 1e6:10.1f} us/op")

    per_call_2 = bench_forward(demos=2, iters=2000)
    print(f"forward end-to-end (2 demos, dummy LM)   {per_call_2 * 1e6:10.1f} us/op")

    wall, ideal, score = bench_orchestration()
    print(
        f"eval 256 x 20ms @ 16 threads             {wall * 1e3:10.1f} ms wall "
        f"(ideal {ideal * 1e3:.0f} ms, overhead {(wall - ideal) * 1e3:.1f} ms, score {score})"
    )
