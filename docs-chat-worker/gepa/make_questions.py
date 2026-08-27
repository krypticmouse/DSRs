"""Teacher-generated question expansion for the docs-chat eval pool.

Walks the docs source (docs/docs/**/*.mdx, skipping the auto-generated
rustdoc index pages under api/) and asks a teacher model to write new
questions a real DSRs user might ask that each page answers. Every row is
validated locally (schema, near-duplicate token overlap against the existing
50 questions and other generated rows, key-point support against the page
text) before it is accepted.

    .venv/bin/python make_questions.py                # ~90 accepted rows
    .venv/bin/python make_questions.py --dry-run      # page plan, no API calls

Output: new_questions.jsonl, same row schema as train.jsonl:
    {"question": str, "key_points": [str], "citations": [<page slug>]}

Keys come from the repo-root .env (OPENROUTER_API_KEY).
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run import HERE, load_env

DOCS_ROOT = HERE.parents[1] / "docs" / "docs"
TEACHER_MODEL = "openrouter/openai/gpt-5.6-terra"
EXISTING_FILES = ("train.jsonl", "val.jsonl", "holdout.jsonl")
OUT = HERE / "new_questions.jsonl"

STOPWORDS = set(
    "the a an of to in is are and or for with on by as at it its this that "
    "be from what how do does can when which why not you your i".split()
)

PROMPT = """You are building an evaluation set for a documentation assistant \
for DSRs (dspy-rs, a Rust framework for building and optimizing LM pipelines).

Below is ONE documentation page. Write {n} NEW questions that a real DSRs \
user might ask and that THIS page answers.

Requirements for each question:
- Self-contained: a user typing it into a chat box, with no reference to \
"this page" or "the docs above".
- Realistic: the kind of question a Rust developer using DSRs would actually \
ask (API shape, behavior, errors, persistence, configuration, gotchas).
- Answerable from this page alone.
- NOT a duplicate or near-duplicate of any question in the existing list \
below. Cover material the existing questions do not.

Requirements for key_points (3 to 6 per question):
- Each is an independently checkable factual claim stated by this page.
- Specific: exact type names, method names, variants, defaults, field names.
- Non-overlapping: no two key points restate the same fact.
- Written as standalone declarative sentences, like these examples from the \
existing set:
  "PredictError has four variants: Lm, Parse, Conversion, and Replay."
  ".call() returns Result<Predicted<S::Output>, PredictError>."

## Existing questions (do not duplicate)
{existing}

## Documentation page: {slug}
{page}

Return ONLY a JSON object:
{{"questions": [{{"question": "...", "key_points": ["...", "..."]}}, ...]}}"""


def iter_pages():
    """Substantive pages only: skip api/ (auto-generated rustdoc link tables,
    never cited by the existing evalset) and any index page."""
    for path in sorted(DOCS_ROOT.rglob("*.mdx")):
        rel = path.relative_to(DOCS_ROOT)
        if rel.parts[0] == "api" or path.stem == "index":
            continue
        yield path


def slug_of(path):
    # docs/docs/components/predict.mdx -> "components/predict"
    return str(path.relative_to(DOCS_ROOT).with_suffix("")).replace("\\", "/")


def questions_for(path):
    """Bigger pages support more non-overlapping questions."""
    size = path.stat().st_size
    if size >= 15000:
        return 5
    if size < 5000:
        return 2
    return 3


def tokens(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]


def near_duplicate(question, others):
    """Token-overlap near-duplicate check: Jaccard or containment on content
    tokens. Returns the offending existing question, or None."""
    a = set(content_tokens(question))
    if not a:
        return "empty question"
    for other in others:
        b = set(content_tokens(other))
        if not b:
            continue
        inter = len(a & b)
        jaccard = inter / len(a | b)
        containment = inter / min(len(a), len(b))
        if jaccard >= 0.5 or containment >= 0.8:
            return other
    return None


def supported(key_point, page_token_set):
    """Fuzzy sanity check (not an LLM): enough of the key point's distinctive
    tokens must literally appear in the page text."""
    distinctive = [t for t in content_tokens(key_point) if len(t) >= 4]
    if len(distinctive) < 3:
        return True  # too short to judge lexically; keep
    hits = sum(t in page_token_set for t in distinctive)
    return hits / len(distinctive) >= 0.6


def call_teacher(slug, page_text, n, existing_block):
    import litellm

    resp = litellm.completion(
        model=TEACHER_MODEL,
        temperature=0.7,
        max_tokens=8000,
        response_format={"type": "json_object"},
        num_retries=2,
        drop_params=True,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    n=n, existing=existing_block, slug=slug, page=page_text
                ),
            }
        ],
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    return json.loads(text).get("questions", [])


def validate(cand, slug, page_token_set, accepted_questions, existing_questions, rejects):
    q = cand.get("question")
    kps = cand.get("key_points")
    if not isinstance(q, str) or not q.strip() or not isinstance(kps, list):
        rejects.append((slug, "bad-schema", str(cand)[:80]))
        return None
    q = q.strip()
    kps = [k.strip() for k in kps if isinstance(k, str) and k.strip()]
    dup = near_duplicate(q, existing_questions)
    if dup:
        rejects.append((slug, "near-dup-existing", f"{q[:60]} ~ {dup[:60]}"))
        return None
    dup = near_duplicate(q, accepted_questions)
    if dup:
        rejects.append((slug, "near-dup-generated", f"{q[:60]} ~ {dup[:60]}"))
        return None
    kept = [k for k in kps if supported(k, page_token_set)]
    for k in set(kps) - set(kept):
        rejects.append((slug, "unsupported-key-point", k[:80]))
    if not 3 <= len(kept) <= 6:
        rejects.append((slug, "too-few-key-points", f"{q[:60]} ({len(kept)} kept)"))
        return None
    return {"question": q, "key_points": kept, "citations": [slug]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the page plan, no API calls")
    ap.add_argument("--workers", default=4, type=int)
    args = ap.parse_args()

    pages = list(iter_pages())
    plan = [(p, slug_of(p), questions_for(p)) for p in pages]
    print(f"{len(plan)} pages, {sum(n for _, _, n in plan)} questions requested")
    if args.dry_run:
        for _, slug, n in plan:
            print(f"  {n}  {slug}")
        return

    load_env()

    existing_questions = []
    for name in EXISTING_FILES:
        for line in (HERE / name).read_text().splitlines():
            if line.strip():
                existing_questions.append(json.loads(line)["question"])
    print(f"{len(existing_questions)} existing questions loaded")
    existing_block = "\n".join(f"- {q}" for q in existing_questions)

    def run_page(task):
        path, slug, n = task
        try:
            return slug, call_teacher(slug, path.read_text(), n, existing_block), None
        except Exception as e:  # noqa: BLE001 — one bad page must not kill the run
            return slug, [], f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        raw = list(ex.map(run_page, plan))

    # validate serially in deterministic slug order so the generated-vs-
    # generated duplicate check is reproducible
    page_text = {slug_of(p): p.read_text() for p in pages}
    accepted, rejects, errors = [], [], []
    for slug, cands, err in sorted(raw, key=lambda r: r[0]):
        if err:
            errors.append((slug, err))
            continue
        page_token_set = set(tokens(page_text[slug]))
        for cand in cands:
            row = validate(
                cand,
                slug,
                page_token_set,
                [r["question"] for r in accepted],
                existing_questions,
                rejects,
            )
            if row:
                accepted.append(row)

    OUT.write_text("".join(json.dumps(r) + "\n" for r in accepted))
    print(f"\naccepted {len(accepted)} rows -> {OUT}")
    by_reason = {}
    for _, reason, _ in rejects:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    print(f"rejected: {by_reason or 'none'}")
    for slug, reason, detail in rejects:
        print(f"  [{reason}] {slug}: {detail}")
    for slug, err in errors:
        print(f"  [page-error] {slug}: {err}")


if __name__ == "__main__":
    main()
