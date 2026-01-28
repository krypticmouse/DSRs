
# Tests: Collapsing the Remaining Space

Types decide what states you can even talk about.  
Traits decide what capabilities can exist.  
Errors decide how those capabilities can refuse.

**Tests decide which implementations of that story we will actually allow to exist.**

Types/traits/errors still leave you with an enormous space of possible programs that are “technically valid.”  
A test suite’s job is simple:

> **Collapse that remaining space until only a handful of implementations survive, ideally one.**

No more, no less.

---

## 0. Forcing functions

Keep these four rules in your head whenever you write or review a test.

1. **Every test must kill a family of wrong implementations.**  
   If you can’t point at a *non‑contrived* implementation that:
   - passes all existing tests,  
   - fails *this* one,
   - and would be bad in production,  
   then the test is noise.

2. **Tests are executable attempts to falsify claims.**  
   A test is always of the form:

   > “Under this model of the world, for these inputs, this relation between before/after must hold.”

   No claim → no test.

3. **Assert each fact at the lowest level where it can be expressed cleanly.**  
   If something can be stated and checked:
   - on a pure function instead of an HTTP endpoint,  
   - on a trait implementation instead of a full system,  
   do it there.  
   E2E is the last place a fact should appear, not the first.

4. **A test that doesn’t change a decision surface is decoration.**  
   If this behavior regressed, what concrete decision would change?  
   - An HTTP code?  
   - A retry vs no‑retry?  
   - A permission check?  
   - A money move?  
   If you can’t answer, delete or rewrite the test.

---

## 1. What a test is (in our universe)

We treat each test as:

- a **claim** – a sentence you could write in English,
- a **search space** – which inputs / worlds it actually explores,
- an **oracle** – a predicate that says “the claim held here.”

Mechanically:

```text
Claim:     decode(encode(x)) == x  for all valid x
Space:     100 random x + a few hand-picked edge cases
Oracle:    assert_eq!(decode(encode(x)), x)
````

If you can’t name all three, you don’t understand the test yet.

---

## 2. The only shapes of tests we use

To shrink the design space, we pretend there are only four kinds of tests.
Every test must be exactly one of these:

1. **Law tests (`∀`)**

   * “For all valid inputs, property P holds.”
   * Usually property‑based or small, carefully‑chosen sets of examples.
   * Lives at the level of types and traits.
   * Examples:

     * once `Iterator::next` returns `None`, it never returns `Some`;
     * scoring is monotone in some dimension;
     * decode(encode(x)) = x.

2. **Example tests (`∃`, canonical)**

   * “For this specific, meaningful input, this output / error happens.”
   * These are the examples you’d put in docs.
   * Good for weird business rules, boundary cases, and explaining semantics.

3. **Scenario tests (protocol/flow)**

   * “Given this starting state and this series of actions, these observable outcomes happen.”
   * Cross‑component behavior: HTTP → app → DB, queues, etc.
   * Focused on externally visible behavior and contracts.

4. **Regression tests (fossilized bug)**

   * “This *exact* combination of inputs and world used to break. Never again.”
   * Named after incidents / tickets when possible.
   * Ideally get absorbed into a law or example test once you understand the underlying rule.

We don’t distinguish “unit/integration/e2e” by ceremony; those are just scopes:

* A “unit test” is a **law or example** test at a small boundary.
* An “integration test” is a **scenario** over more parts of the system.

If you can’t tag a test as Law / Example / Scenario / Regression, change it until you can.

---
## 2.1 Slow tests are opt-in

Some property tests exercise expensive setup (git repos, IPC daemons, large payloads). These are
valuable but **must not** run by default.

Guidelines:
- Gate heavy suites behind `--features slow-tests`.
- Default `cargo test` should stay fast and predictable.
- CI should include at least one job that runs slow suites (scheduled or manual).

Run locally when needed:

```bash
cargo test --features slow-tests
```

Totally fair. That section *was* a bit hand‑wavy for how central contracts are in your world.

Here’s a beefed‑up **Section 3** you can drop in verbatim over the old one — still mechanical, but with enough shape that people will actually build the right thing.

## 3. Contract tests for traits

From the trait doc: traits are promises made to strangers.  
Contract tests are how we **enforce** those promises across implementations.

Mechanically:

> For every cold trait, we want *one* shared contract suite that every implementation must pass.

If a trait has no contract suite, it’s polymorphism without accountability.

---

### 3.1 Shape of a contract suite

For a cold trait `T`, we define:

1. A **factory** for implementations used in tests:

   ```rust
   pub trait Make<T> {
       fn make(&self) -> T;
   }
   ```

or just `impl Fn() -> T` where that’s enough.

2. A set of **law tests** that take a `&Make<T>` (or `impl Fn() -> T`) and only talk in terms of the trait:

   ```rust
   pub fn storage_laws<S, M>(m: &M)
   where
       S: Storage,           // the trait under test
       M: Make<S>,           // how to get a fresh instance
   {
       get_after_put_round_trips(m);
       put_is_idempotent(m);
       errors_match_spec(m);
   }
   ```

3. Optional **scenario tests** for protocol‑level guarantees that live at the trait boundary:
   concurrency behavior, ordering, time‑related semantics, etc.

Key constraint:

> Contract tests talk only in terms of the trait’s types, methods, and documented error enums.
> They never reach into implementation‑specific fields, DB schemas, or helper methods.

If a test needs to look behind the trait, that behavior doesn’t belong in the trait contract.

---

### 3.2 Example: `Storage`

Suppose we have:

```rust
pub trait Storage {
    type Key;
    type Value;
    type Error;

    fn get(&self, key: &Self::Key) -> Result<Option<Self::Value>, Self::Error>;
    fn put(&self, key: Self::Key, value: Self::Value) -> Result<(), Self::Error>;
}
```

A minimal contract suite might look like:

```rust
pub fn storage_laws<S, M>(m: &M)
where
    S: Storage<Key = String, Value = String, Error = StorageError>,
    M: Make<S>,
{
    get_after_put_returns_value(m);
    put_overwrites_previous_value(m);
    get_on_missing_key_is_none(m);
    put_is_idempotent_for_same_value(m);
    invalid_key_surfaces_as_invalid_key_error(m);
}
```

Each helper:

* gets a **fresh** `S` from the factory,
* exercises the trait only through `get`/`put`,
* asserts on:

  * returned values,
  * `StorageError` variants and classification helpers,
  * documented side effects (e.g. overwrites vs merges).

When you add a new backend (`InMemoryStorage`, `PostgresStorage`, `RemoteStorage`), you just:

```rust
#[test]
fn postgres_storage_satisfies_storage_laws() {
    let maker = PostgresStorageMaker::new(test_db_url());
    storage_laws::<PostgresStorage, _>(&maker);
}
```

If it can’t pass, there are only two possibilities:

1. The implementation is wrong; or
2. The trait’s promise doesn’t match reality and needs to change (split trait, adjust errors, etc.).

Both are useful outcomes.

---

### 3.3 Which traits get contracts?

We only bother with full contract suites for **cold traits**:

* used across modules/crates,
* expected to have multiple implementations *or* to survive for a long time,
* whose misuse would be expensive (storage, evaluation, auth, payments…).

Hot traits (experiments, local abstractions) can live with local tests.
Once a trait goes cold, the contract suite is required.

---

### 3.4 Rules of thumb

When designing/using contract tests:

* **One suite per trait**, not per implementation.
  The suite lives next to the trait definition, not next to the backends.

* **Single source of truth for laws.**
  The laws written in the trait docs and the laws encoded in the contract tests should be the same set of statements.

* **Cold by default.**
  Contract tests are cold: changing them means changing the trait’s spec. That should be rare and deliberate.

* **No “implementation‑only” tests that duplicate the contract.**
  Implementation‑specific tests should focus on things *beyond* the shared contract
  (e.g. performance guarantees, migration behavior), not re‑assert the same laws.

Traits without contract tests are polymorphism without accountability.
Implementations that don’t run the contract suite are untrusted.

---

## 4. Hot vs cold tests

We reuse the temperature idea:

* **Cold tests**

  * Encode behavior you’re committing to long‑term: trait laws, protocol contracts, public HTTP semantics.
  * These should rarely change. Breaking them is a spec change, not a refactor.

* **Hot tests**

  * Surround experimental code, unstable APIs, in‑progress refactors.
  * They’re allowed to be looser and more example‑oriented.
  * They should either mature into cold tests or be deleted.

Rules:

* Don’t pin hot behavior with cold tests.
* Don’t run hot tests in the same gates / pipelines as cold ones without labeling; otherwise everything feels equally sacred.

---

## 5. Smells (short list)

Things that should trigger an immediate “why does this exist?” reaction:

1. **Coverage‑driven noise**

   * Tests whose only purpose is bumping coverage.
   * Names don’t describe a claim; assertions are tautologies (“it returns something”).
   * Ask: *which family of wrong implementations does this kill?* If you can’t answer, delete.

2. **Interaction fetish**

   * Tests asserting “method X called method Y once” via mocks.
   * The behavior you actually care about is at the boundary (what got returned / written / emitted).
   * Fix: assert on observable outcomes, not call graphs, unless the call pattern *is* the spec.

3. **Assertion on non‑spec details**

   * Tests that fail because of:

     * log message wording,
     * JSON field order,
     * internal error strings,
     * implementation details you never promised.
   * Fix: assert on semantic things:

     * error *variants* or classes,
     * HTTP status + structured payload,
     * DB state / side effects.

4. **Facts only asserted at the top**

   * Big E2E tests checking business rules that could be checked on a small pure function or trait.
   * When they fail, you have no idea where the bug is.
   * Fix: move the rule down to the smallest layer that can express it; keep only a thin smoke test up top.

---

## 6. Tiny checklist

When you write or review a test, you should be able to answer these quickly:

1. **What’s the claim?**
   One sentence, in domain language.

2. **What’s the shape?**
   Law / Example / Scenario / Regression — exactly one.

3. **What’s the search space?**
   Single example, a few edge cases, generator, recorded trace?

4. **What’s the oracle?**
   What exactly are we asserting, and is it about *semantics* or *incidental details*?

5. **Which family of wrong implementations does this kill?**
   Name at least one real, plausible “bad implementation” that this test would catch and that other tests wouldn’t.

6. **Is this the lowest level we can assert this fact?**
   If not, move it down.

7. **Is this hot or cold?**
   Are we willing to treat this as “spec” or is it scaffolding around moving targets?

If you can’t answer these, you’re not designing tests yet — you’re just writing code that happens to live in `tests/`.

---

## 7. What tests are *for* in this codebase

Given the rest of our design philosophy:

* Types constrain what states can exist.
* Traits constrain what we can do with those states.
* Errors constrain how failure is expressed.

**Tests constrain which implementations of all that are allowed to ship.**

Practically, that means:

* We write **law suites** for cold traits and core types.
* We add a small number of **example** tests to pin weird domain behavior.
* We use **scenario** tests to enforce contracts at HTTP / job / integration boundaries.
* We add **regression** tests only when a real incident proved we were wrong about something.

And every time we add a test, we ask the same question:

> “Which bad program am I ruling out by doing this?”

If the answer is “none that I care about,” the correct move is to not write the test.
