#!/usr/bin/env python3
"""Agentic docs + code search backed by Mixedbread's toast-1.

Two stores are synced: `dsrs-docs` holds the published docs
(docs/index.mdx, docs/docs/**, docs/snippets/**) and `dsrs-code` holds
the Rust sources (crates/**/*.rs). Queries run against both with
agentic search / grep hosted tools, which routes retrieval through
toast-1.

Usage (from the repository root, needs `pip install mixedbread` and
MXBAI_API_KEY in the environment or in .env):

    python3 docs/scripts/search_docs.py sync [--dry-run]
    python3 docs/scripts/search_docs.py search "how do optimizers mutate holes?"
    python3 docs/scripts/search_docs.py ask "what is a hole?"

Sync is incremental: files are keyed by repo-relative path (external_id)
and skipped when their sha256 matches the store's copy; files deleted
from disk are removed from the store. CI runs sync on pushes to main.
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

# store -> (base dir, glob patterns relative to it)
STORES = {
    "dsrs-docs": (DOCS, ["index.mdx", "docs/**/*.mdx", "snippets/**/*.mdx"]),
    "dsrs-code": (REPO / "crates", ["**/*.rs"]),
}


def load_api_key():
    if os.environ.get("MXBAI_API_KEY"):
        return
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key == "MXBAI_API_KEY" and value:
                os.environ["MXBAI_API_KEY"] = value.strip().strip("\"'")
                return
    sys.exit("MXBAI_API_KEY is not set (env or .env)")


def client():
    load_api_key()
    try:
        from mixedbread import Mixedbread
    except ImportError:
        sys.exit("missing dependency: pip install mixedbread")
    return Mixedbread()


def local_files(base, patterns):
    files = {}
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            rel = str(path.relative_to(base))
            files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def store_files(mxbai, store):
    files, after = {}, None
    while True:
        page = mxbai.stores.files.list(store, limit=100, after=after)
        for f in page.data:
            files[f.external_id or f.filename] = f
        if len(page.data) < 100:
            return files
        after = page.data[-1].id


def sync_store(mxbai, store, base, patterns, dry_run):
    try:
        mxbai.stores.retrieve(store)
    except Exception:
        print(f"creating store {store}")
        if not dry_run:
            mxbai.stores.create(name=store)

    local = local_files(base, patterns)
    try:
        remote = store_files(mxbai, store)
    except Exception:
        remote = {}

    stale = [f for rel, f in remote.items() if rel not in local]
    changed = [
        rel
        for rel, digest in local.items()
        if rel not in remote or (remote[rel].metadata or {}).get("sha256") != digest
    ]

    def upload(rel):
        print(f"upload {store}:{rel}")
        if dry_run:
            return
        # explicit content type: the server's sniffing mislabels .mdx/.rs
        mime = "text/markdown" if rel.endswith((".mdx", ".md")) else "text/plain"
        mxbai.stores.files.upload_and_poll(
            store_identifier=store,
            file=(Path(rel).name, (base / rel).read_bytes(), mime),
            external_id=rel,
            overwrite=True,
            metadata={"path": rel, "sha256": local[rel]},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(upload, changed))
    for f in stale:
        print(f"delete {store}:{f.external_id or f.filename}")
        if not dry_run:
            mxbai.stores.files.delete(f.id, store_identifier=store)
    print(f"{store}: {len(changed)} uploaded, {len(stale)} deleted, "
          f"{len(local) - len(changed)} unchanged")


def sync(args):
    mxbai = client()
    for store, (base, patterns) in STORES.items():
        sync_store(mxbai, store, base, patterns, args.dry_run)


def search(args):
    mxbai = client()
    stores = [args.store] if args.store else list(STORES)
    results = mxbai.stores.search(
        query=args.query,
        store_identifiers=stores,
        top_k=args.top_k,
        search_options={"agentic": True, "return_metadata": True},
    )
    if args.json:
        print(json.dumps([c.model_dump(mode="json") for c in results.data], indent=2))
        return
    for chunk in results.data:
        path = chunk.external_id or chunk.filename
        print(f"── {path}  (score {chunk.score:.3f})")
        print(getattr(chunk, "text", "") or "")
        print()


def ask(args):
    """Grounded answer from toast-1 itself, not just retrieved chunks."""
    import urllib.request

    load_api_key()
    stores = list(STORES)
    body = json.dumps({
        "model": "toast-1",
        "messages": [{"role": "user", "content": args.query}],
        "tools": [
            {"type": "store_search", "store_identifiers": stores},
            {"type": "store_grep", "store_identifiers": stores},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.mixedbread.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['MXBAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.load(resp)
    print(result["choices"][0]["message"]["content"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="sync docs + code into the stores")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.set_defaults(func=sync)

    p_search = sub.add_parser("search", help="agentic search over docs + code")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=8)
    p_search.add_argument("--store", choices=list(STORES), default=None,
                          help="restrict to one store (default: all)")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=search)

    p_ask = sub.add_parser("ask", help="grounded toast-1 answer over docs + code")
    p_ask.add_argument("query")
    p_ask.set_defaults(func=ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
