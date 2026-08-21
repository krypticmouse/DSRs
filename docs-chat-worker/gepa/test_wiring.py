"""Wiring test: full GEPA loop with a mocked student and judge.

Costs a few cents (the reflection LM is real — it must be, to test that
'openrouter/openai/gpt-5.6-sol' works as a reflection_lm string) but makes
zero Mixedbread calls. The mock judge rewards prompts that mention citing,
so a working loop should discover a prompt containing 'cite' and beat the
seed's 0.35 val score.

    python test_wiring.py
"""

import run as runmod

runmod.load_env()

import gepa

import adapter
import judge as judgemod


def fake_student(system_prompt, question):
    has_cite = "cite" in system_prompt.lower()
    return f"stub answer; prompt_mentions_citing={has_cite}", [
        {"type": "store_search_call", "queries": [question]}
    ]


def fake_judge(question, answer, key_points, citations):
    if "prompt_mentions_citing=True" in answer:
        return 0.85, "Good: cites sources.", {}
    return 0.35, "Answer never cites file paths. Instruct the assistant to cite sources.", {}


adapter.call_student = fake_student
adapter.judge = fake_judge
judgemod.judge = fake_judge

trainset = [
    {"question": f"q{i}", "key_points": ["kp"], "citations": []} for i in range(4)
]

result = gepa.optimize(
    seed_candidate={"system_prompt": "You are the DSRs docs assistant. Answer concisely."},
    trainset=trainset,
    valset=trainset,
    adapter=adapter.DocsChatAdapter(max_workers=2),
    reflection_lm="openrouter/openai/gpt-5.6-sol",
    reflection_minibatch_size=2,
    max_metric_calls=30,
    display_progress_bar=False,
    seed=0,
)

print("candidates explored:", result.num_candidates)
print("val scores:", [round(s, 2) for s in result.val_aggregate_scores])
print("best prompt:", repr(result.best_candidate["system_prompt"][:200]))
assert result.val_aggregate_scores[result.best_idx] > 0.35, "loop never improved on seed"
print("WIRING OK")
