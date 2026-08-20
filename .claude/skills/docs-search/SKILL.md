---
name: docs-search
description: Agentic search over the DSRs docs via Mixedbread toast-1. Use when answering questions about DSRs concepts, components, or APIs (signatures, predict, modules, IR, holes, optimizers, .dsrs files) instead of grepping docs/ page by page.
---

# DSRs docs search

The published docs are indexed in a Mixedbread Store (`dsrs-docs`) and
the Rust sources in `dsrs-code`; both are queried with toast-1 agentic
search, which decomposes the question into subqueries and returns
curated chunks. `ask` can also grep the code store for exact symbols.

Run from the repository root (needs `mixedbread` installed and
`MXBAI_API_KEY` in the environment or `.env`):

```bash
python3 docs/scripts/search_docs.py search "<question>" [--top-k N] [--json]
python3 docs/scripts/search_docs.py ask "<question>"
```

`search` returns raw chunks with source paths — prefer it when you plan to
read the pages yourself. `ask` has toast-1 compose a grounded answer —
prefer it for a quick factual check.

- Phrase the query as a full question, not keywords — toast-1 plans its
  own subqueries from it.
- Each result prints the source page path (e.g.
  `docs/components/holes.mdx`) and the chunk text; expect ~10s latency.
- Read the source page under `docs/` when a chunk looks truncated.
- If results look stale relative to the working tree, the index only
  tracks pushed docs — trust the local files and optionally run
  `python3 docs/scripts/search_docs.py sync` to refresh.
