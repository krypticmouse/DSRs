# Persistent interaction storage

`InteractionStore` persists invocation records independently of the response
cache, bounded inspection history and trace capture. `JsonlInteractionStore`
ships with the core crate and requires neither `data` nor `cache` features.

```rust,no_run
use std::sync::Arc;
use dspy_rs::{InteractionStore, JsonlInteractionStore, with_interaction_store};
# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let store = Arc::new(JsonlInteractionStore::open("interactions.jsonl")?);
let result = with_interaction_store(store.clone(), async {
    // Call predictors, composed modules or LM::call here.
}).await?;
let recent = store.recent(20)?;
if let Some(entry) = recent.first() {
    let same_call = store.get(&entry.id)?;
}
# Ok(()) }
```

Successful typed predictions expose the invocation IDs in
`prediction.metadata().call_ids`; direct LM calls expose their record through
`response.history`. Agent predictions include IDs for each LM round. IDs are
unique for cache hits too. Replay-only execution creates no new LM invocations.
Cache identity continues to include generation settings; storage never replaces
or supplies cache entries.

Run `capture(|| module.call(input))` inside the storage scope to persist the
complete execution trace as well, including IR tool results and non-LM spans.
Completed captures are saved automatically; retrieve them with
`store.get_trace(&trace.meta.trace_id)`. Invocation `trace_id` fields join LM
records to their enclosing trace, and predictions retain their span IDs. Nested
capture semantics are unchanged: the innermost capture owns its spans. A scope
cancelled before capture finishes still retains cancelled LM invocation records,
but has no completed trace to save.

The scope records raw completions, parsed output or parse errors, provider errors
and dropped invocations. Parsing updates the original ID. Each record contains
redacted configuration, input messages, tool events, usage and module path.
Credentials are redacted from configuration; prompt/response content is retained.

Nested storage scopes compose. Spawned tasks must establish their own scope.
The outer `Result` reports persistence failures; an inner predictor/LM `Result`
still reports execution failures. Cancelling the entire scope cannot return an
error, so storage errors during cancellation are also emitted through tracing.

The journal uses OS file locks across handles/processes and syncs every revision.
It recovers an incomplete final line before appending and rejects corrupt complete
records or unknown versions. Reads return the latest revision per ID, and
reopening a store preserves records. Writes are synchronous, and reads/updates
scan the journal: this backend suits modest local workloads. The trait supports
application-specific indexed database backends. Configure retention/archival
outside the journal; bounded in-memory history does not delete stored records.
