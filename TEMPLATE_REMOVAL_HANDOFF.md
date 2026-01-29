# Template System Removal - Handoff Context

## What We're Doing

Removing the "typed template system" from the codebase. This system was added to allow Jinja-based custom rendering of signature inputs/outputs, but it turned out to be useless because:

1. By the time templates run, we only have `BamlValue` (type info lost)
2. `DefaultJinjaRender` trait exists but was never wired into the rendering pipeline
3. The `#[render = "..."]` field attribute was **never used** by any code
4. Complex types like `REPLHistory` still need to be pre-rendered to String anyway

## The Surgery

We're rebasing all commits after the template system onto the commit BEFORE templates were added.

### Key Commits

```
upvvptxv (8c1acfba) - "wip: document LM configuration"     <- REBASE TARGET (before templates)
    |
wwwryrkw (63a42b7e) - "macros: add template/render attrs"  <- TEMPLATE START (skip this)
urxmyuyy (89179f9a) - "Add system/user templates"          <- TEMPLATE (skip this)
yxnkpkrt (6b530853) - "Fix truncate filter"                <- TEMPLATE (skip this)
ysstmvss (6afd6da1) - "Validate signature templates"       <- TEMPLATE (skip this)
zzmvxsym (ce024f00) - "Match default templates"            <- TEMPLATE END (skip this)
    |
... 80 commits ...
    |
pxxpxnrs (current @) - "rlm: align typed loop"             <- CURRENT WORK
```

### The Command

```bash
jj rebase -s 'children(zzmvxsym)' -d 'parents(wwwryrkw)'
```

This says: take all children of `zzmvxsym` (the last template commit) and rebase them onto the parent of `wwwryrkw` (the first template commit), effectively skipping the template commits.

## Expected Conflicts

Only **3 commits** will conflict:

### 1. `ytuwvxnz` - "dsrs-vn6.7.8: fix clippy warnings"

**File:** `crates/dspy-rs/src/adapter/jinja_types.rs`

**Problem:** This file won't exist after rebase (it was added by templates).

**Resolution:** The change was a 1-line clippy fix (`&HashMap` → `HashMap`). Just **abandon this part** of the commit or let jj drop it since the file doesn't exist.

```bash
# If this commit becomes empty or conflicts, just:
jj abandon <change-id>
# Or resolve by accepting that jinja_types.rs doesn't exist
```

### 2. `rrzvquyu` - "wip: some work im not sure we should keep"

**File:** `crates/dsrs-macros/src/lib.rs`

**Problem:** This commit adds `#[signature(rlm = true|false)]` attributes and references `system_template`/`user_template` fields in the `ParsedSignature` struct that won't exist.

**Resolution:** Keep the RLM-related additions, remove references to template fields:

- Keep: `generate_rlm_input: bool` field
- Keep: `#[signature(rlm = ..., rlm_input = ...)]` parsing
- Remove: any references to `system_template`, `user_template`, `system_template_warning_span`, `user_template_warning_span`
- Update error message to not mention template attrs

### 3. `pxxpxnrs` - "rlm: align typed loop with DSPy prompts" (current @)

**Files:** `crates/dspy-rs/src/adapter/chat.rs`, `crates/dspy-rs/src/adapter/jinja_types.rs`

**Problem:** This is our current work that added `try_render_default_template()` to jinja_types.rs (which won't exist) and called it from chat.rs.

**Resolution:** This work was trying to make `REPLHistory` render nicely as a native object. Since we're removing the template system, we should:

1. **Revert signatures to use `String` for `repl_history`** (not `REPLHistory` object)
2. **Remove the `try_render_default_template` function** entirely
3. **Keep `REPLHistory` as an internal type** that gets `.render()`ed to String before passing to signatures

The key files to check in current `@`:
- `crates/dspy-rs/src/rlm/signatures.rs` - revert `repl_history: REPLHistory` back to `repl_history: String`
- `crates/dspy-rs/src/rlm/rlm.rs` - pass `history.render()` instead of `history.clone()`

## What Gets Removed (Template System)

These files/code will no longer exist after rebase:

### Files Deleted
- `crates/dspy-rs/src/adapter/jinja_types.rs` (JinjaField, JinjaBamlValue)
- `crates/dspy-rs/src/adapter/templates.rs` (DEFAULT_*_TEMPLATE constants)
- `crates/dspy-rs/tests/test_typed_templates.rs`
- `crates/dspy-rs/tests/test_default_template_format.rs`
- `crates/dspy-rs/tests/test_default_template_schema_golden.rs`

### Code Removed from chat.rs
- `build_template_context_base()`
- `add_template_inputs()` / `add_template_outputs()`
- `render_template()`
- `render_system_message_typed()` / `render_user_message_typed()` / `render_assistant_message_typed()`
- Imports: `minijinja`, `JinjaField`, `JinjaBamlValue`, `DEFAULT_*_TEMPLATE`

### Code Removed from dsrs-macros
- `#[render = "..."]` field attribute parsing
- `#[signature(system_template = "...", user_template = "...")]` parsing
- `system_template()` / `user_template()` methods on Signature trait
- Template validation at compile time (minijinja AST parsing)
- `proc-macro-warning` dependency

### Dependencies Removed
- `minijinja` from dspy-rs (BUT check if RLM still needs it for REPLHistory!)
- `indoc` from dspy-rs
- `proc-macro-warning` from dsrs-macros

## What Stays (RLM Still Needs minijinja)

**IMPORTANT:** The RLM module uses minijinja for `REPLHistory::render()` via the `DefaultJinjaRender` trait. This is SEPARATE from the typed template system.

Check these files to see if minijinja is still needed:
- `crates/dspy-rs/src/rlm/history.rs` - `REPLHistory::render()`
- `crates/baml-bridge/src/render_trait.rs` - `DefaultJinjaRender` trait

If RLM still needs minijinja, keep it in Cargo.toml but remove the template-specific usage.

## Recovery Commands

```bash
# See operation history
jj op log

# Undo the rebase entirely
jj undo

# Restore to a specific operation
jj op restore <op-id>

# See what a conflicted commit looks like
jj log -r 'conflicts()'

# Resolve conflicts in a commit
jj new <conflicted-change>  # check it out
# ... edit files ...
jj squash                   # fold resolution into the conflicted commit
```

## Verification After Rebase

```bash
# 1. Check for conflicts
jj log -r 'conflicts()'

# 2. Verify template files are gone
ls crates/dspy-rs/src/adapter/jinja_types.rs  # should not exist
ls crates/dspy-rs/src/adapter/templates.rs    # should not exist

# 3. Build and test
cargo build -p dspy-rs
cargo test -p dspy-rs

# 4. Check that RLM still works (if minijinja kept)
cargo build -p dspy-rs --features rlm
```

## End State

After successful rebase:
1. No `jinja_types.rs` or `templates.rs`
2. No template methods in `chat.rs` (reverted to pre-template formatting)
3. No template attrs in dsrs-macros
4. `Signature` trait has no `system_template()` / `user_template()` methods
5. RLM works with `REPLHistory` rendered to String before passing to signatures
6. All tests pass

## Context From Discussion

The user (darin) was frustrated because:
> "it means that all of the template system is literally pointless"

The template system looked like it should help with custom type rendering, but it fundamentally couldn't because type information is lost when converting to `BamlValue`. The `#[render]` attribute and custom templates could only manipulate the serialized JSON data, not call methods on the original Rust types.

The decision: remove it entirely rather than maintain dead code.
