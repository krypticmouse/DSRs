"""Same task, fields, token budget and success criterion as bench-target-22.rs."""
import json
import os
import sys
import time

import dspy
from dspy.clients import configure_cache
from dspy.utils.dummies import DummyLM


class QA(dspy.Signature):
    """Answer the geography question with only the city name in the answer field."""
    question: str = dspy.InputField(desc="The geography question.")
    answer: str = dspy.OutputField(desc="The city name, without extra commentary.")


mode = sys.argv[1]
assert mode in ["live", "cached", "offline"]
iterations, warmup = (1, 1) if mode == "live" else (1000, 100)
configure_cache(enable_disk_cache=False, enable_memory_cache=True)
adapter = dspy.JSONAdapter()


class LocalSchemaLM(dspy.LM):
    # Explicit capability opt-in for this tested custom endpoint, as in Rust.
    @property
    def supports_response_schema(self):
        return True

    @property
    def supported_params(self):
        return super().supported_params | {"response_format"}


if mode == "offline":
    lm = DummyLM([{"answer": "Paris"}] * (iterations + warmup), adapter=adapter)
else:
    lm = LocalSchemaLM("openai/llama3.2", api_base=os.environ["OLLAMA_BASE_URL"],
                 api_key="local-not-a-secret", temperature=0.0, max_tokens=64,
                 cache=mode == "cached", num_retries=0)
dspy.configure(lm=lm, adapter=adapter, disable_history=True)
predictor = dspy.Predict(QA)
for _ in range(warmup):
    assert predictor(question="What is the capital of France?").answer == "Paris"
samples = []
for _ in range(iterations):
    started = time.perf_counter_ns()
    output = predictor(question="What is the capital of France?")
    samples.append(time.perf_counter_ns() - started)
    assert output.answer == "Paris"
print(json.dumps({"framework": "dspy", "version": dspy.__version__, "mode": mode,
                 "iterations": iterations, "warmup": warmup, "samples_ns": samples, "answer": "Paris"}))
