## What an AGENTS.md is

An `AGENTS.md` file is **executable context that makes a limited-visibility contributor behave correctly**.

It's not documentation. It's a **behavioral forcing function** that encodes:
- What this code is and isn't allowed to be
- How to work here without breaking invariants
- Which patterns to copy and which to avoid
- How to verify you didn't screw up

These files form a **hierarchy that mirrors your directory tree**. As an agent (or human) navigates to a specific file, they accumulate context:

```
/AGENTS.md                     → "This is how we think about software in this repo"
/server/AGENTS.md              → "...and this server specifically handles X, never Y"  
/server/routes/AGENTS.md       → "...and routes follow pattern P, verify with command C"
/server/routes/vapi/AGENTS.md  → "...and vapi has this specific gotcha G"
```

Each level **refines** but never **contradicts** the levels above. By the time you're editing `/server/routes/vapi/webhooks.rs`, you have the full gradient from philosophy → domain rules → local tactics.

---

## The AGENTS.md Spec

Each `AGENTS.md` has three moves:

### 1. BOUNDARY
*What this directory IS and ISN'T.*

```markdown
## Boundary
This directory: [single sentence of purpose]
Depends on: [what it's allowed to import/call]
Depended on by: [who can import/call this]  
NEVER: [what must not leak out or happen here]
```

This prevents architectural drift. The agent knows what belongs here vs elsewhere.

### 2. BEHAVIOR  
*How to work here correctly.*

```markdown
## How to work here
Golden example: [path/to/best_example.rs - what to copy]
When adding X: [concrete steps including verification]
Tests that matter: `[exact command that must pass]`
Invariants: [what must always remain true]
```

This is the oral tradition. The agent knows what good looks like and how to check it didn't break things.

### 3. ENTROPY CONTROL
*What not to do, even if you see it in the code.*

```markdown
## Don't copy this
Legacy pattern: [what exists but shouldn't be copied]
Use instead: [the modern pattern]
Known issues: [gotchas that will waste time]
```

This prevents the agent from mimicking the worst parts of your codebase.

---

## Hierarchy rules

1. **Root `/AGENTS.md`** sets universal laws. These are axioms—unbreakable, apply everywhere.
2. **Domain `/{subsystem}/AGENTS.md`** adds domain rules. These specialize but don't contradict root.
3. **Leaf `/{subsystem}/{module}/AGENTS.md`** adds local tactics. Maximum specificity, minimum size.

Each file is **small** (< 100 lines). Dense signal, no essays. Links to deeper docs if needed.

When an agent works in `/foo/bar/baz.rs`, it sees all AGENTS.md files from root to `baz/`, stacked. This gives it the full context gradient without needing to see the whole repo.

---

## Why this works

This spec has stupidly high utility because:

1. **Boundary** stops the agent from putting code in the wrong place (the #1 source of architectural rot)
2. **Behavior** gives the agent a concrete pattern to copy + a way to verify it worked (the Factory thesis)
3. **Entropy Control** stops the agent from propagating the bad parts that already exist
4. **Hierarchy** means you write each rule once at the right level, not 50 times

An agent following this spec will write code that looks like a senior engineer who's been on the team for years wrote it. Not because the agent is smart, but because you've installed your taste directly into its context window at exactly the right level of abstraction for where it's working.

That's what an AGENTS.md is: **a context patch that turns a talented stranger into a trusted teammate**.

===

cool, let's “spec” this thing.

Below is a *general* AGENTS.md spec you can reuse anywhere, built from everything we’ve been iterating on.

---

# AGENTS.md – Informal Spec

## 0. Core definition (one-liner)

An `AGENTS.md` is a **hierarchical, executable guide** for a slice of the repo that encodes:

- What this code **is for** (boundary & responsibilities)
- How we **work here** (patterns, style, workflows)
- How we **verify** we didn’t break anything (commands/tests/invariants)
- What we **never do** here (negative space / anti‑patterns)

Stacked across the directory tree, these files give an agent (or human) enough *worldview + tactics* to behave like a long‑tenured contributor, despite only seeing a tiny local window of code.

---

## 1. Hierarchy semantics

**1.1 Where they live**

- Any directory MAY contain an `AGENTS.md`.
- The root of the repo SHOULD contain one.
- Subtrees that define clear domains (e.g., `server/`, `core/`, `ingest/`, `tests/`) SHOULD contain one at their root.

**1.2 How they stack**

When working on a file at path:

```text
/path/to/repo/a/b/c/file.ext
```

the agent loads, in order:

1. `/AGENTS.md`              (root)
2. `/a/AGENTS.md`           
3. `/a/b/AGENTS.md`
4. `/a/b/c/AGENTS.md`

These are conceptually concatenated (or merged) into a single “context patch.”  
Deeper levels **refine / specialize**, but should not silently **contradict** higher ones.

**1.3 Precedence**

- Root defines **global axioms** (non‑negotiable truths).
- Subdirectories define **domain doctrine** (how those axioms apply in this area).
- Leaf directories define **local tactics / gotchas / workflows**.

If there is a real conflict, higher level wins, and the lower‑level file should eventually be edited to remove the conflict.

---

## 2. High-level responsibilities of an AGENTS.md

Each file should:

1. **Define the boundary** – what this directory is for, what depends on it, and what it depends on.
2. **Install a worldview** – the local taste, principles, and heuristics that drive decisions here.
3. **Encode verifiability** – how to cheaply check that code changes here are correct.
4. **Prescribe patterns** – how to structure code/tests/config in this subtree.
5. **Constrain entropy** – what must never be done or extended here, even if older code does it.
6. **Point to deeper context** – links to design/type/architecture docs that matter for this area.
7. **Stay small and operational** – high signal, no essays; deep philosophy lives elsewhere.

---

## 3. Recommended section layout

You don’t have to literally name them this way, but conceptually every AGENTS.md should cover:

1. `## Boundary` – what lives here and the dependency rules
2. `## How to work here` – patterns, code style, and workflows
3. `## Verification` – commands/tests you *must* run and what they guarantee
4. `## Don’t do this` – legacy patterns and forbidden moves
5. `## Gotchas` – traps, caveats, perf issues, non‑obvious stuff
6. `## References` – links to deeper docs

Root and domain levels will be heavier; leaf levels can be much lighter.

---

## 4. Section semantics (detailed)

### 4.1 `## Boundary`

**Goal:** make the architectural role of this directory explicit.

Include:

- **Purpose (1–2 sentences)**
  - “This directory implements X and nothing else.”
  - Avoid vague crap; name the domain and responsibility.

- **Dependencies (allowed imports/calls)**
  - “May depend on: `core/types`, stdlib.”
  - “Must NOT depend on: HTTP, DB, external services.”

- **Dependents (who can call/use this)**
  - “Used by: `server/routes`, `cli/`.”
  - This is useful both for humans and for automated refactor tools.

- **Non‑leak invariants**
  - “Provider‑specific schemas must not appear outside this folder.”
  - “Database types must not escape this layer.”
  - “No business logic here; this is mapping/transport only.”

This is your **boundary contract**. It answers: “What belongs here vs elsewhere?”

---

### 4.2 `## How to work here`

This is the main “oral tradition” section. It’s what you wish a competent stranger knew before touching anything.

Sub‑pieces:

#### 4.2.1 Golden patterns

- **Point to canonical files**:
  - “For a canonical handler pattern, see `foo/bar.rs`.”
  - “For a clean property-based test, see `tests/roundtrip.rs`.”

- Optionally show a **tiny code snippet** (inline) if the pattern is non‑obvious.

The goal is: when the agent needs to add X, it knows *exactly what to imitate*.

#### 4.2.2 Code style & structure (for this area)

High‑signal conventions that *actually matter* here:

- **Language-level style**
  - “Prefer small, pure functions; avoid 200‑line methods.”
  - “Don’t use `unwrap`/`expect` in production code here.”
  - “Explicit enums over stringly‑typed variants.”

- **Error handling patterns**
  - “All errors go through `Error` in `errors.rs` (use `thiserror`).”
  - “Bubble errors up as `Result<T, MyError>`, never `anyhow::Error` in this module.”

- **Data modeling patterns**
  - “Lifecycle is represented via typestates `Foo<Draft>`/`Foo<Final>`; don’t add boolean flags.”
  - “Keep raw provider payloads alongside canonical types (no silent lossy transforms).”

- **Local architectural style**
  - “Routes are thin; they call into services in `services/`, which call into repositories in `repo/`.”
  - “Config is read at the edge and passed in; don’t reach for globals.”

- **Formatting / linting specifics if they impact semantics**
  - “We use `rustfmt` defaults; don’t fight it.”
  - “Run `ruff` with config X; new code should follow that style automatically.”

Do **not** re-explain what the linter already enforces unless it’s subtle or controversial. Focus on *taste that the tools can’t easily enforce*.

#### 4.2.3 Local workflows

- **How you actually work here when making changes**
  - “To add a new provider mapping:
    1. Copy `foo_provider.rs`.
    2. Update the schema types.
    3. Wire it into the registry in `mod.rs`.
    4. Run `cargo test -p ingest new_provider_roundtrip`.”

- **How to debug**
  - “To debug weird trace ordering issues, log X and run Y.”
  - “Use this CLI tool to replay a webhook against this code.”

Basically: “When I (human you) need to touch this area, what do I actually do?”

---

### 4.3 `## Verification`

This is the Factory talk made concrete: how we validate work in this subtree.

Include:

- **Required commands**
  - Exact shell commands:
    - `cargo test -p my_crate ingest_roundtrip`
    - `pytest tests/ingest/test_vapi.py`
    - `just check:server`
  - Make them copy‑pastable and scoped; don’t always say “run every test in the repo.”

- **What they guarantee**
  - “This ensures provider payloads map to canonical traces without panicking.”
  - “This checks that all HTTP routes still follow the contract.”

- **When they must be run**
  - “Run this whenever you:
    - Add a new provider.
    - Touch mapping logic.
    - Change trace types.”
  - VS. “Only needed for larger refactors; small comment changes don’t require this.”

Optional but nice:

- **Non‑test validators**
  - “If you change migrations, run `sqlx prepare --check`.”
  - “Run `cargo clippy -p core` for new public APIs.”

The agent should be able to finish its edits, run the commands listed here, and know: “I am probably safe.”

---

### 4.4 `## Don’t do this`

This is entropy control / negative space.

Include:

- **Legacy patterns not to copy**
  - “You will see usage of `OldFixtureFactory`; don’t use it. Use `NewTestHarness` instead.”
  - “Old code may parse JSON by hand; new code must use typed adapters.”

- **Forbidden operations**
  - “No direct DB access from this layer.”
  - “No new global env vars; thread config through function parameters.”
  - “Do not introduce new public fields to `CoreType` without updating its docs/tests.”

- **Banned style (where needed)**
  - “No `unwrap()`/`expect()` here except in tests.”
  - “No `anyhow::Error` in this module; use specific error types.”

This is where you explicitly say: “Even if it compiles and there’s precedent in old code, *don’t* do it.”

---

### 4.5 `## Gotchas`

Short list of things that have bitten you before:

- Flaky tests and how to think about them.
- Env quirks (“needs REDIS_URL set or tests will hang”).
- Non‑obvious performance cliffs.
- Dangerous partial refactors (“if you change X but not Y, everything still compiles but behavior is wrong”).

Keep this tight. Each bullet should be something you wish previous‑you had known, not random trivia.

---

### 4.6 `## References`

Pointers, not walls of text:

- Deep design docs:
  - `/docs/types/principles.md`
  - `/docs/architecture/ingest.md`
- External specs / API docs:
  - “Provider X webhook schema: <url>”
- Local glossaries if you have jargon (“trace”, “span”, “segment” etc).

This is how you connect the small operational view to your big philosophical essays.

---

## 5. Root vs Domain vs Leaf specifics

### 5.1 Root `/AGENTS.md` (Constitution)

Contains:

- **Global worldview / taste (very compact)**
  - 3–7 bullets that define how this repo thinks about software:
    - “Types should tell the truth, especially uncomfortable truths.”
    - “Information is never silently dropped; lossy transforms are explicit.”
    - “Rust is source of truth; Python is legacy/delete‑only.”

- **Repo topology**
  - What the major top-level directories are for.
  - Maybe a tiny map: `core/`, `server/`, `ui/`, `scripts/`, `tests/`.

- **Global axioms / bans**
  - E.g., “No `unwrap`/`expect` in production code anywhere.”
  - “Raw external payloads must always be storable and introspectable.”

- **Global verification norm**
  - A couple of commands you basically always run before pushing to main.
  - High‑level testing story (unit vs integration vs e2e).

This file is rarely changed and has the highest level of abstraction.

### 5.2 Domain `AGENTS.md` (subsystem roots)

For directories like `core/`, `ingest/`, `server/`, `tests/`, etc.

Heavier on:

- Boundary: domain responsibility and deps.
- Local taste: how this domain interprets the global principles.
- Good patterns/bad patterns specific to this domain.
- Domain‑specific verification (e.g., property tests in core, e2e tests in server).

This is where most of the interesting content lives.

### 5.3 Leaf `AGENTS.md` (deep modules)

Very focused; often:

- A 1–2 sentence boundary remark if needed.
- Exact commands to run when touching this.
- 1–2 golden exemplar file paths.
- 1–2 don’ts / gotchas.

Think of these as “micro‑playbooks” for the really weird/dense areas.

---

## 6. Code style: where it belongs

Code style has two homes:

1. **Global style (root-level)**
   - Language‑wide norms:
     - Rust: prefer enums + typestates; no `unwrap`; explicit errors.
     - Python: type hints required in `app/`; ruff config is canonical style.
   - Formatting rules:
     - “Always run `rustfmt` / `ruff format` / `prettier` on save.”
   - High‑level patterns:
     - “Keep functions short; long ones must be heavily structured.”
     - “Prefer composition over inheritance.”
   - If your linter config already enforces most of this, just:
     - link to it; and
     - note any *non-linted* taste you care about.

2. **Local style (per-domain AGENTS.md)**
   - Only include style rules that are *domain‑specific*:
     - Test style in `/tests/`.
     - Error mapping style in `/ingest/`.
     - API response shaping style in `/server/`.

Don’t duplicate linter manuals. Focus on how style + architecture + domain semantics interact.

---

## 7. Writing guidelines

A few meta‑rules for authoring AGENTS.md files:

- **Write for a smart stranger with local eyesight.**
  - Assume they can read code and tests.
  - Assume they can search the repo.
  - Don’t assume they know the history or tribal rules.

- **Be concrete.**
  - Prefer “run `cargo test -p ingest roundtrip`” over “run the tests.”
  - Prefer “copy `foo.rs`’s pattern” over “follow good FP practices.”

- **Keep them small but dense.**
  - Aggressively cut fluff.
  - If something is important but long, move it to `docs/` and link it.

- **Don’t repeat yourself across levels.**
  - Put global stuff in root.
  - Put domain stuff in domain root.
  - Leaf files should specialize, not re-explain.

- **Update with behavior.**
  - Whenever you realize “AGENTS.md lied to me” during a change:
    - update it in the *same PR* to reflect reality.
  - Treat these files like code, not like static docs.

---

## 8. Tooling expectations (for later)

If you (or a platform) wire this into an agent:

- For a given edit operation, the agent should:
  - Load **all applicable AGENTS.md files** along the path.
  - Use them to:
    - shape its system prompt / instructions;
    - decide which commands/tests to run after editing;
    - decide what patterns to look at in the repo as few‑shot examples.
- CI / bots can:
  - Run the commands specified under `## Verification` for touched directories.
  - Enforce banned patterns under `## Don’t do this`.

---

If a contributor (human or LLM) **actually follows** a well-written AGENTS.md that obeys this spec, you get:

- Architecture that doesn’t silently rot.
- Style that stays coherent without bikeshedding.
- Changes that are verifiable with cheap, local loops.
- Agents that behave like opinionated seniors, not autocomplete with a GPU.

That’s the bar.

