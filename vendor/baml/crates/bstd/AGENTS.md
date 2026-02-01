# bstd - BAML Standard Utilities

## Boundary

**Purpose:** Vendored BAML standard library providing small, self-contained utilities.

**This is vendored code.** Modifications should be extremely rare and only for critical fixes.

**Provides:**
- `dedent` / `DedentedString` - String dedentation with indent tracking
- `ProjectFqn` - Fully qualified project name parsing (`org/project` format)
- `random_word_id()` - Tailscale-style random identifiers (`fox-lizard-123`)
- `pluralize()` - Simple singular/plural selection

**Dependencies:** `anyhow`, `num`, `rand`, `regex`

## How to Work Here

**You probably shouldn't.** This is vendored upstream BAML code.

If you must modify: document why upstream can't be used, keep changes minimal, consider upstreaming.

## Verification

```bash
cargo test -p bstd
```

## Don't Do This

- Don't add new utilities here - put DSR-specific utilities elsewhere
- Don't refactor or "improve" this code - it's vendored for stability
- Don't update dependencies without understanding upstream compatibility

## Gotchas

- `random_word_id()` uses Tailscale's wordlist - IDs are SFW and unoffensive by design
- `ProjectFqn` enforces specific naming rules: org allows `_` prefix, project must start with lowercase letter
