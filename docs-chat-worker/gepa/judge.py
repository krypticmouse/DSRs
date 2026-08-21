"""LLM judge for docs-chat answers: key-point coverage minus noise.

The teacher model grades an answer against a per-question checklist and
returns a scalar score plus textual feedback. The feedback string is what
GEPA's reflection step consumes, so it must name what was missed and quote
what was fluff — a bare number wastes the optimizer's main advantage.

Score design (coverage-dominant, so the optimizer cannot reward-hack):
  0.70 * key-point coverage      -- terse-but-lossy answers lose here
  0.10 * required citations      -- file paths the answer should mention
  0.20 * (1 - noise fraction)    -- verbose answers lose here
  -0.15 per contradicted fact (capped at 0.30)
"""

import json
import os

import litellm

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openrouter/openai/gpt-5.6-sol")

JUDGE_PROMPT = """\
You are grading an answer from a documentation assistant for DSRs (dspy-rs, \
a Rust framework for building and optimizing LM pipelines).

## Question
{question}

## Answer under evaluation
{answer}

## Key points a correct answer must contain
{key_points}

## File paths the answer should cite (empty = no citation required)
{citations}

Grade strictly. "Covered" means the fact is stated correctly, not merely \
alluded to. "Fluff" means spans that carry no key point: preamble, restating \
the question, hedging, generic filler, or detail nobody asked for.

Return ONLY a JSON object with these fields:
{{
  "covered": [one boolean per key point, in the same order as listed],
  "missed": ["each absent or wrong key point, restated briefly"],
  "fluff": ["verbatim spans from the answer that carry no key point"],
  "citations_present": ["paths from the required list that the answer mentions"],
  "wrong_claims": ["claims that contradict the key points, if any"],
  "advice": "2-5 sentences of direct advice to the assistant: what to add, what to cut, how to restructure."
}}"""


def judge(question, answer, key_points, citations):
    """Returns (score: float in [0,1], feedback: str, raw: dict)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer or "(empty answer)",
        key_points="\n".join(f"{i+1}. {k}" for i, k in enumerate(key_points)),
        citations="\n".join(citations) if citations else "(none required)",
    )
    resp = litellm.completion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        num_retries=2,
    )
    raw = _parse_json(resp.choices[0].message.content)

    covered = raw.get("covered", [])
    coverage = sum(bool(c) for c in covered) / max(1, len(key_points))
    cite = (
        len(raw.get("citations_present", [])) / len(citations) if citations else 1.0
    )
    fluff_chars = sum(len(s) for s in raw.get("fluff", []))
    noise = min(1.0, fluff_chars / max(1, len(answer or "")))
    wrong_penalty = min(0.30, 0.15 * len(raw.get("wrong_claims", [])))

    score = 0.70 * coverage + 0.10 * min(1.0, cite) + 0.20 * (1.0 - noise)
    score = max(0.0, min(1.0, score - wrong_penalty))

    feedback = _compose_feedback(raw, coverage, noise, score)
    return score, feedback, raw


def _compose_feedback(raw, coverage, noise, score):
    lines = [
        f"Score {score:.2f} (coverage {coverage:.0%}, noise {noise:.0%} of answer)."
    ]
    if raw.get("missed"):
        lines.append("Missing or wrong:")
        lines += [f"  - {m}" for m in raw["missed"]]
    if raw.get("wrong_claims"):
        lines.append("Contradicts the docs:")
        lines += [f"  - {w}" for w in raw["wrong_claims"]]
    if raw.get("fluff"):
        lines.append("Fluff to cut:")
        lines += [f'  - "{f}"' for f in raw["fluff"][:5]]
    if raw.get("advice"):
        lines.append(f"Advice: {raw['advice']}")
    return "\n".join(lines)


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    return json.loads(text)
