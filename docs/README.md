# Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mint) to preview your documentation changes locally. To install, use the following command:

```
npm i -g mint
```

Run the following command at the root of your documentation, where your `docs.json` is located:

```
mint dev
```

View your local preview at `http://localhost:3000`.

## Need help?

### Troubleshooting

- If your dev environment isn't running: Run `mint update` to ensure you have the most recent version of the CLI.
- If a page loads as a 404: Make sure you are running in a folder with a valid `docs.json`.

### Resources
- [Mintlify documentation](https://mintlify.com/docs)
- [Mintlify community](https://mintlify.com/community)

## API Reference tab

The pages under `docs/api/` are generated from rustdoc JSON by
`docs/scripts/gen_api.py`; do not edit them by hand. To regenerate after a
public-API change, run from the repository root:

```bash
RUSTC_BOOTSTRAP=1 cargo rustdoc -p dspy-rs --lib --all-features -- -Z unstable-options --output-format json
RUSTC_BOOTSTRAP=1 cargo rustdoc -p dsrs-tools --lib -- -Z unstable-options --output-format json
RUSTC_BOOTSTRAP=1 cargo rustdoc -p dsrs_macros --lib -- -Z unstable-options --output-format json
python3 docs/scripts/gen_api.py
```

## Agentic search (toast-1)

The published pages are indexed in a Mixedbread Store (`dsrs-docs`) and
the Rust sources (`crates/**/*.rs`) in `dsrs-code`; both are searchable
through [toast-1](https://www.mixedbread.com/blog/toast-1) agentic
retrieval, which also greps the code store for exact symbol lookups.
Needs `pip install mixedbread` and `MXBAI_API_KEY` (env or repo-root
`.env`):

```bash
python3 docs/scripts/search_docs.py search "how do optimizers mutate holes?"
python3 docs/scripts/search_docs.py ask "what is a hole?"   # grounded answer
python3 docs/scripts/search_docs.py sync   # re-index local docs changes
```

CI re-syncs the store on pushes to `main` that touch docs content
(`.github/workflows/docs-search-sync.yaml`), using the `MXBAI_API_KEY`
repository secret.

## AI chat widget

`docs/toast-chat.js` is auto-injected into every page by Mintlify and adds
an "Ask AI" chat backed by toast-1, grounded in the `dsrs-docs` store. The
browser never sees the API key: requests go through the Cloudflare Worker
in `docs-chat-worker/`. To deploy it:

```bash
cd docs-chat-worker
npx wrangler deploy
npx wrangler secret put MXBAI_API_KEY
```

Then set the deployed URL in `docs/toast-chat.js` (the
`YOUR-SUBDOMAIN.workers.dev` placeholder) and tighten `ALLOWED_ORIGINS` in
`wrangler.toml` to the docs domain. For local preview, `npx wrangler dev
--port 8787` in `docs-chat-worker/` — the widget targets
`localhost:8787` automatically when running under `mint dev`.
