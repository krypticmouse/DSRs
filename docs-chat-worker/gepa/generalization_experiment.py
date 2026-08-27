"""Does the marker protocol's refusal suppression generalize beyond toast-1?

Inline-context QA (no doc stores needed): 6 questions whose answer is absent
from the provided context (correct behavior = say so) + 6 answerable controls
(correct behavior = answer; checks the refusal-safe sentence doesn't cause
over-refusal). Three arms, identical instruction (which already grants
refusal permission — same configuration as the toast-1 finding):

  bare — instruction as system, labeled raw inputs as user
  old  — ChatAdapter rendering before the 2026-08 template fix
         (no refusal sentence, 8-space indented objective)
  new  — ChatAdapter rendering after the fix

Students: mainstream models via OpenRouter. Judge: gpt-5.6-sol classifies
each answer as refused / fabricated / hedged (unanswerable) or
answered / refused / wrong (answerable).

    python generalization_experiment.py     # ~330 short calls, a few $
"""

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm

from run import load_env

HERE = Path(__file__).parent
REPEATS = 3
WORKERS = 8
STUDENTS = [
    "anthropic/claude-sonnet-5",
    "google/gemini-3.7-flash",
    "meta-llama/llama-4-maverick",
]
JUDGE = "openrouter/openai/gpt-5.6-sol"

INSTRUCTION = (
    "Answer the question using only the provided context. If the context does "
    "not contain the information needed, say the context does not cover it "
    "instead of guessing. Answer in concise markdown."
)

OLD_RESPONSE = (
    "Respond with the corresponding output fields, starting with the field "
    "`[[ ## answer ## ]]`, and then ending with the marker for "
    "`[[ ## completed ## ]]`."
)
NEW_RESPONSE = (
    OLD_RESPONSE
    + " If the information needed for an output field is unavailable, say so "
    "in that field instead of guessing."
)

STRUCTURE = """Your input fields are:
1. `context` (string): Reference material for answering.
2. `question` (string): The question to answer.

Your output fields are:
1. `answer` (string)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## context ## ]]
context

[[ ## question ## ]]
question

[[ ## answer ## ]]
Output field `answer` should be of type: string

[[ ## completed ## ]]"""


def old_system():
    indented = "".join(f"\n        {line}" for line in INSTRUCTION.splitlines())
    return (
        f"{STRUCTURE}\n\n{OLD_RESPONSE}\n\n"
        f"In adhering to this structure, your objective is: {indented}"
    )


def new_system():
    return (
        f"{STRUCTURE}\n\n{NEW_RESPONSE}\n\n"
        f"In adhering to this structure, your objective is:\n{INSTRUCTION}"
    )


def rendered_user(item, response_line):
    return (
        f"[[ ## context ## ]]\n{item['context']}\n\n"
        f"[[ ## question ## ]]\n{item['question']}\n\n{response_line}"
    )


def bare_user(item):
    return f"context:\n{item['context']}\n\nquestion:\n{item['question']}"


ARMS = {
    "bare": lambda item: (INSTRUCTION, bare_user(item)),
    "old": lambda item: (old_system(), rendered_user(item, OLD_RESPONSE)),
    "new": lambda item: (new_system(), rendered_user(item, NEW_RESPONSE)),
}


def strip_markers(text):
    s = text.split("[[ ## answer ## ]]")[-1]
    return s.split("[[ ## completed ## ]]")[0].strip()


JUDGE_PROMPT = """You are grading whether a QA system handled a context-grounded question correctly.

Context given to the system:
---
{context}
---
Question: {question}
Ground truth: {truth}

System's answer:
---
{answer}
---

Classify the answer as exactly one label:
- "refused": the answer states the context does not cover / does not contain the needed information (possibly with a brief note of what the context does cover).
- "answered": the answer gives a substantive answer to the question drawn from the context.
- "fabricated": the answer gives a substantive answer that is NOT supported by the context (invented facts).
- "hedged": the answer half-answers with heavy uncertainty without clearly refusing or clearly answering.

Reply with JSON only: {{"label": "..."}}"""


def judge_one(item, answer):
    truth = (
        f"NOT ANSWERABLE from the context. {item.get('note', '')}"
        if not item["answerable"]
        else f"Answerable; key fact: {item['key_fact']}"
    )
    resp = litellm.completion(
        model=JUDGE,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    context=item["context"],
                    question=item["question"],
                    truth=truth,
                    answer=answer or "(empty)",
                ),
            }
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    text = text.split("{", 1)[-1].rsplit("}", 1)[0]
    return json.loads("{" + text + "}")["label"]


def call_student(model, system, user):
    resp = litellm.completion(
        model=f"openrouter/{model}",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


def main():
    load_env()
    os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
    items = json.loads((HERE / "generalization_data.json").read_text())
    tasks = [
        (model, arm, rep, idx, item)
        for model in STUDENTS
        for arm in ARMS
        for rep in range(REPEATS)
        for idx, item in enumerate(items)
    ]
    print(f"{len(items)} questions x {len(ARMS)} arms x {REPEATS} reps x {len(STUDENTS)} models = {len(tasks)} rollouts")

    def run_one(task):
        model, arm, rep, idx, item = task
        try:
            system, user = ARMS[arm](item)
            raw = call_student(model, system, user)
            answer = strip_markers(raw) if arm in ("old", "new") else raw.strip()
            label = judge_one(item, answer)
            return {
                "model": model, "arm": arm, "rep": rep, "q_idx": idx,
                "answerable": item["answerable"], "label": label,
                "chars": len(answer),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "model": model, "arm": arm, "rep": rep, "q_idx": idx,
                "answerable": item["answerable"], "label": None,
                "error": str(e)[:200],
            }

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, tasks):
            results.append(r)
            done += 1
            if done % 30 == 0:
                print(f"progress {done}/{len(tasks)}", flush=True)

    out = HERE / "generalization_results.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    failed = sum(1 for r in results if r["label"] is None)
    print(f"\nfailed rollouts: {failed}")
    cells = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r["label"] is None:
            continue
        key = (r["model"], r["arm"], "unanswerable" if not r["answerable"] else "answerable")
        cells[key][r["label"]] += 1
        cells[key]["n"] += 1

    print(f"\n{'model':<32} {'arm':<5} | unanswerable: refusal%  fabricated% | answerable: answered%  refused%")
    for model in STUDENTS:
        for arm in ("bare", "old", "new"):
            ua = cells[(model, arm, "unanswerable")]
            aa = cells[(model, arm, "answerable")]
            if not ua.get("n") or not aa.get("n"):
                continue
            print(
                f"{model:<32} {arm:<5} | "
                f"{100 * ua.get('refused', 0) / ua['n']:>8.0f}%  "
                f"{100 * ua.get('fabricated', 0) / ua['n']:>10.0f}% | "
                f"{100 * aa.get('answered', 0) / aa['n']:>9.0f}%  "
                f"{100 * aa.get('refused', 0) / aa['n']:>7.0f}%"
            )
    print(f"\nraw results in {out}")


if __name__ == "__main__":
    main()
