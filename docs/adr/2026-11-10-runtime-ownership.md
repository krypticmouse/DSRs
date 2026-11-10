# Runtime ownership

DSRs separates immutable program descriptions, live runtime bindings and mutable
invocation state. Sharing follows these boundaries rather than cloning a whole
program to change its model.

| Data | Owner and sharing rule | Lifetime |
| --- | --- | --- |
| Typed signature metadata | One immutable allocation per Rust signature type; adapters borrow its fields | Process, bounded by compiled types |
| Dynamic signatures and program graph | `Arc<Program>` owned by the caller and loaded interpreters; no global dynamic-program cache | Last owner drops |
| Compiled system prompt | Interpreter-owned cache, at most one instruction/prompt pair per signature; changing an instruction replaces the entry | Loaded interpreter |
| Provider, HTTP client and prebuilt tool set | Shared handles; each interpreter resolves its own model/tool bindings | Owning LM/interpreter |
| Predictor instruction, demos and overlay | Owned configuration or immutable scoped candidate; changing a candidate does not mutate another rollout | Predictor or evaluation scope |
| Input, Jinja context, parse maps and output | Owned by one invocation; ordinary fields render by borrowing, one alias-aware Jinja value serves all templated fields | Formatting or invocation |
| Trace and recorded content | Capture owns its recording; serialized traces are explicit caller-owned artifacts | Capture/caller |

`Interpreter::load(Program, env)` takes ownership and wraps the graph once.
`Interpreter::load_shared(Arc<Program>, env)` validates and binds an existing
graph. Both run identical validation and capability checks. A predictor that
rebinds its LM uses `load_shared`; rebinding no longer deep-clones the graph.
Each run still owns its input/output map, overlay view, metering and trace state.
Sharing a graph never shares a run's mutable state or replaces capability checks.

System-prompt caching keys on the signature ID within this immutable program
and compares the complete resolved instruction. A changed instruction rebuilds
the prompt. Live inputs, demos, playbooks and feedback are rendered or appended
per call and never retained by that cache. Concurrent instruction scopes may
replace the last-used entry but cannot change one another's returned prompt.

Typed signature metadata can be process-lived because the set of Rust types is
fixed. Dynamic input strings and loaded graphs must not be leaked to obtain
`'static` references. Graph node IDs reference their owning program; callers must
bind and validate overlays against that same program.

An LM clone shares provider and cache handles. This is intentional connection
and cache reuse, not independent configuration construction. Configuration copies
and independent history lifecycles are specified in the LM configuration/history
PRs; graph ownership does not silently redefine those APIs.

Remaining value copies have semantic reasons: materializing a typed input at the
IR boundary, taking an owned recorded response, constructing provider-owned HTTP
messages and exposing an independently owned result. Optimizing those copies
requires a measured workload and preserved lifetime/API contracts. This decision
does not require eliminating every `clone()` in the repository.

`test_program_ownership` checks pointer identity, concurrent independent model
bindings and release after the final owner drops. `test_jinja_context` checks
fresh per-call values, aliases, no input mutation and recovery after render
errors. The controlled rendering comparison records allocation and latency
effects separately; it does not attribute a speedup to graph sharing.
