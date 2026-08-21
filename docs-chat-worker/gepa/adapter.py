"""GEPA adapter for the docs-chat worker.

Candidate = {"system_prompt": <text>}: the one component GEPA evolves,
deployed by pasting the winner into SYSTEM in ../worker.js.

evaluate() runs each eval question through toast-1 (same store tools as the
worker) and scores the answer with the judge. Trajectories keep the searches
toast-1 ran (hosted_tool_calls) so the reflection step can see retrieval
behavior, not just the final answer.
"""

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from gepa.core.adapter import EvaluationBatch

from judge import judge

MXBAI_URL = "https://api.mixedbread.com/v1/chat/completions"
STORES = ["dsrs-docs", "dsrs-code"]
MAX_TOKENS = 2048  # matches the deployed worker


def call_student(system_prompt, question, attempts=3):
    """One toast-1 rollout with retry. Returns (answer_text, hosted_tool_calls)."""
    for attempt in range(attempts):
        try:
            return _call_student_once(system_prompt, question)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def _call_student_once(system_prompt, question):
    req = urllib.request.Request(
        MXBAI_URL,
        data=json.dumps(
            {
                "model": "toast-1",
                "stream": False,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "tools": [
                    {"type": "store_search", "store_identifiers": STORES},
                    {"type": "store_grep", "store_identifiers": STORES},
                ],
            }
        ).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['MXBAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    answer = resp["choices"][0]["message"].get("content") or ""
    return answer, resp.get("hosted_tool_calls") or []


def _summarize_tool_calls(tool_calls):
    out = []
    for tc in tool_calls:
        kind = tc.get("type", "tool_call")
        detail = tc.get("queries") or tc.get("pattern") or ""
        out.append(f"{kind}: {detail}")
    return out


class DocsChatAdapter:
    # gepa probes this attribute; None = use the built-in instruction proposer
    propose_new_texts = None

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["system_prompt"]

        def run_one(item):
            # Per-example failures score 0.0 instead of raising, per the
            # GEPAAdapter contract — one bad rollout must not kill the run.
            try:
                answer, tool_calls = call_student(system_prompt, item["question"])
                score, feedback, _ = judge(
                    item["question"],
                    answer,
                    item["key_points"],
                    item.get("citations", []),
                )
            except Exception as e:
                answer, tool_calls, score = "", [], 0.0
                feedback = f"ROLLOUT ERROR (scored 0): {e}"
            return {
                "question": item["question"],
                "answer": answer,
                "tool_calls": _summarize_tool_calls(tool_calls),
                "score": score,
                "feedback": feedback,
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(run_one, batch))

        return EvaluationBatch(
            outputs=[r["answer"] for r in results],
            scores=[r["score"] for r in results],
            trajectories=results if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [
            {
                "Inputs": {"question": t["question"]},
                "Generated Outputs": {
                    "answer": t["answer"],
                    "searches_run": t["tool_calls"],
                },
                "Feedback": t["feedback"],
            }
            for t in eval_batch.trajectories
        ]
        return {"system_prompt": records}
