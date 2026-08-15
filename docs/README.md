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
