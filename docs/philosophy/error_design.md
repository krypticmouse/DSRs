# Errors: The Other Half of the Map

Types tell one story: what success looks like.  
Traits tell another: what capabilities exist and how they compose.

Errors are the missing half of the map:
**how those capabilities actually fail in the real world.**

If types are the coordinates of valid states, errors are the geography of refusal:

- where the world said “no,”
- what state we ended up in instead,
- and what moves are still possible from there.

This is not a framework. There’s no global error type you have to funnel everything through.
It’s a way of thinking plus a handful of concrete patterns so good error designs fall out of you by reflex.

Baselines:

- Use `Result<T, E>` for operations that can fail.
- `E` is a **structured domain error**, not a `String` or `anyhow::Error` (except at very outer edges).
- Panics are for **bugs**. If the user, network, disk, clock, or another process can cause it, it’s not a panic.
- In Rust, we use:
  - `thiserror` for enums that implement `std::error::Error`.
  - An **internal error-set type** (currently `terrors::OneOf`) to compose sets of leaf errors inside a module.

Everything else is detail.

---

## 0. Forcing functions

Keep these three in your head; they’re the “snap-to-grid” for everything else:

1. **Design for the caller’s decision surface.**  
   Every error exists to change what the caller does next.  
   If two failures never induce different behavior, they are the same variant.  
   If callers can’t tell failures apart but should, your error type is lying.

2. **Preserve the confession until a conscious boundary.**  
   Inside the system, errors are structured evidence: what we tried, on what, what stopped us.  
   Don’t crush that into a string, HTTP status, or log line until you’re at a boundary where humans or other systems consume it.

3. **Push what you can into types, what you must into errors, and what remains into panics.**  
   Preconditions that must always hold → types.  
   Failures that can happen in a valid world → `Result`.  
   Anything left is a bug; crash loudly.

---

## 1. What an error actually is

Not “something went wrong.”

> **An error is a structured, expected refusal.**  
> “Given those inputs and this world, this operation declined, and here’s the state we ended up in instead.”

- **Success**: we kept the promise, here’s the result.
- **Error**: the request was valid, but the world said no, and we can describe how.
- **Bug**: a precondition or invariant we promised ourselves is false.

Informally, for:

```rust
pub fn op(input: Input) -> Result<Output, Error>
````

read:

> For every `input` that satisfies our preconditions, either:
>
> * we return a valid `Output`, **or**
> * we return an `Error` describing a real state of the world
>   from which a caller can make a meaningful decision.

If there is no meaningful decision to make (“we’re corrupt, crash”), that’s not an error; that’s a bug.

---

## 2. Every error type is a micro‑protocol

From the trait doc: *“Traits are promises made to strangers.”*
Error types are the same thing, but for the negative space.

Erase all code; keep only:

```rust
pub fn create_user(...) -> Result<UserId, CreateUserError>;
```

If the `CreateUserError` docs + variants don’t let a stranger answer:

* Can I safely retry?
* Is this my fault, the domain’s rules, or the environment?
* Should I show a message, change input, or page someone?

…then the error type isn’t done yet.

A **good** error type is one where, knowing only that enum + docs, you can still write sane handling code.

---

## 3. Three levels of error representation

We use three distinct layers:

1. **Leaf error types** – small structs/enums for specific failure modes.
2. **Capability error enums (canonical)** – one per capability, used in traits and at boundaries.
3. **Internal error sets** – `OneOf<(…)>` style sets used *inside* a module for precision & composition.

### 3.1 Leaf errors: precise and local

These are the “atoms” of failure inside a module.

Example:

```rust
#[derive(Debug, thiserror::Error)]
pub enum InvalidKey {
    #[error("key is empty")]
    Empty,

    #[error("key `{raw}` contains forbidden characters")]
    ForbiddenChars { raw: String },
}

#[derive(Debug, thiserror::Error)]
#[error("backend {backend} is unavailable: {reason}")]
pub struct BackendUnavailable {
    pub backend: &'static str,
    pub reason: String,
    #[source]
    pub source: std::io::Error,
}

#[derive(Debug, thiserror::Error)]
#[error("write for `{key}` may have partially completed")]
pub struct UnknownWriteStatus {
    pub key: String,
    #[source]
    pub source: std::io::Error,
}
```

Guidelines:

* Make them **honest, specific stories**: what we were doing, on what, what stopped us.
* Keep them **local** to a module or crate unless they’re genuinely shared concepts.
* You don’t have to expose them publicly; they can be `pub(crate)`.

### 3.2 Capability error enums: one per capability

For each real capability (usually a trait), there is **one canonical error enum**.

This is what we expose across module boundaries, log on, and/or classify.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error(transparent)]
    InvalidKey(#[from] InvalidKey),

    #[error(transparent)]
    BackendUnavailable(#[from] BackendUnavailable),

    #[error(transparent)]
    UnknownWriteStatus(#[from] UnknownWriteStatus),
}
```

This is the thing you put in trait signatures:

```rust
pub trait Storage {
    fn get(&self, key: &Key) -> Result<Option<Value>, StorageError>;
    fn put(&self, key: &Key, value: &Value) -> Result<(), StorageError>;
}
```

Rules:

* **Variants are behaviors; fields are details.**
  If callers should behave differently, they get different variants.
* **No library names in variant names.**
  `InvalidKey`, not `SerdeError`; `BackendUnavailable`, not `ReqwestError`. Underlying crates go in `#[source]`.
* **This is the type you document.**
  “Here is what storage can refuse to do, and why.”

### 3.3 Internal error sets: `OneOf<(…)>` as plumbing

Inside a module, we sometimes want per‑function precision or to compose several capabilities without inventing a new enum every time.

We use an *internal* error-set type for that (currently `terrors::OneOf`):

```rust
use terrors::OneOf;

type StorageGetError = OneOf<(InvalidKey, BackendUnavailable)>;
type StoragePutError = OneOf<(BackendUnavailable, UnknownWriteStatus)>;
```

Internal implementation can use these:

```rust
fn inner_get(...) -> Result<Option<Value>, StorageGetError> { ... }
fn inner_put(...) -> Result<(), StoragePutError> { ... }
```

At the boundary of the capability, we **collapse back** to `StorageError`:

```rust
impl Storage for MyStorage {
    fn get(&self, ...) -> Result<Option<Value>, StorageError> {
        match inner_get(...) {
            Ok(v) => Ok(v),
            Err(e) => match e.as_enum() {
                terrors::E2::A(invalid) => StorageError::InvalidKey(invalid),
                terrors::E2::B(backend) => StorageError::BackendUnavailable(backend),
            },
        }
    }

    fn put(&self, ...) -> Result<(), StorageError> {
        match inner_put(...) {
            Ok(()) => Ok(()),
            Err(e) => match e.as_enum() {
                terrors::E2::A(backend) => StorageError::BackendUnavailable(backend),
                terrors::E2::B(unknown) => StorageError::UnknownWriteStatus(unknown),
            },
        }
    }
}
```

Conventions:

* **`OneOf` stays internal.**
  Public APIs and traits talk in terms of `StorageError`, `UserRepoError`, `EvalError`, etc., not `OneOf`.
* Use `OneOf` when it saves a bunch of glue enums or gives you useful per‑function error sets.
* Don’t turn the whole codebase into a `OneOf` showcase. It’s a plumbing tool, not the user‑facing story.

---

## 4. Designing good errors

### 4.1 Design from the caller’s point of view

The key rule:

> **Distinct behavior ⇒ distinct variant.
> Same behavior ⇒ one variant, more fields.**

Bad:

```rust
pub enum FetchError {
    Io(std::io::Error),
    Http(reqwest::Error),
    Serde(serde_json::Error),
}
```

Callers can’t tell what to do with this. “Serde error” and “Io error” aren’t world states.

Better:

```rust
#[derive(Debug, Error)]
pub enum FetchUserError {
    #[error("could not reach user service at {endpoint}")]
    Transport {
        endpoint: Url,
        #[source]
        source: std::io::Error,
    },

    #[error("user {id} was not found")]
    NotFound { id: UserId },

    #[error("invalid response from user service at {endpoint}")]
    BadResponse {
        endpoint: Url,
        status: u16,
        body_snippet: String,
    },

    #[error("caller is not authorized to fetch user {id}")]
    Unauthorized { id: UserId },
}
```

* `Transport` → maybe retry.
* `NotFound` → show “user not found.”
* `BadResponse` → alert / internal error.
* `Unauthorized` → auth/UX response.

### 4.2 Represent state, not excuses

“Serde error” is not a world state.
“IO error” is not a world state.

World states are:

* “We couldn’t reach the host.”
* “We couldn’t parse the response body.”
* “We tried to write to disk; we don’t know if it committed.”

Every variant should answer:

1. What were we trying to do?
2. On what?
3. What stopped us?
4. What do we know (or not know) about side effects?

If you can’t answer those, you don’t have a variant yet; you have a log line.

### 4.3 Don’t leak dependencies

Public APIs should not force callers to care about which HTTP client or DB driver you picked.

Bad:

```rust
pub enum CreateUserError {
    Sql(sqlx::Error),
    Redis(redis::Error),
    Http(reqwest::Error),
}
```

Better:

```rust
#[derive(Debug, Error)]
pub enum CreateUserError {
    #[error("username `{username}` is invalid: {reason}")]
    InvalidUsername { username: String, reason: String },

    #[error("user `{username}` already exists")]
    AlreadyExists { username: String },

    #[error("could not persist new user `{username}`")]
    Storage {
        username: String,
        #[source]
        source: sqlx::Error,
    },
}
```

Variants talk in **domain** terms. Crate errors live in `#[source]`.

---

## 5. The error pipeline: detect → capture → enrich → decide → render

Think of error handling as a pipeline:

1. **Detect** – an operation fails (syscall, DB query, parse, etc.).
2. **Capture** – close to the failure, we wrap it in a leaf error that describes what we were doing.
3. **Enrich** – as errors bubble up, callers wrap them in higher‑level errors, adding context.
4. **Decide** – at a boundary, we choose policy: retry, fallback, degrade, crash, surface to user.
5. **Render** – we log, emit metrics, or produce an HTTP/CLI response.

Good designs:

* Use **small, honest error types** internally.
* Use **canonical capability enums** at module and trait boundaries.
* Use **coarse, stable categories** only at the skin (HTTP, CLI, jobs).
* **Log once**, at the boundary where you decide. Interior layers preserve structure; they don’t decide policy.

---

## 6. Classification / geometry (optional layer)

Once you have good error enums, it’s useful to have a **small derived view** to drive policy and observability.

A minimal, practical version:

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ErrorClass {
    InvalidInput,
    NotFound,
    Forbidden,
    Temporary,
    Internal,
}
```

Then:

```rust
impl StorageError {
    pub fn class(&self) -> ErrorClass {
        match self {
            StorageError::InvalidKey(_)        => ErrorClass::InvalidInput,
            StorageError::BackendUnavailable(_) => ErrorClass::Temporary,
            StorageError::UnknownWriteStatus(_) => ErrorClass::Internal,
        }
    }
}
```

At boundaries, this feeds:

* HTTP mapping (`ErrorClass::InvalidInput` → 400, `NotFound` → 404, `Temporary` → 503, `Internal` → 500).
* Retry logic (retry only some classes).
* Metrics (group by `ErrorClass`).

### 6.1 Richer geometry (when we need it)

If/when we need more nuance, we can split `ErrorClass` into three axes:

```rust
pub enum Blame {
    Caller,       // bad input, misuse of API
    Domain,       // valid input, but domain rules block it
    Environment,  // network, disk, other services
    Bug,          // our invariant is false
}

pub enum Transience {
    Permanent, // retry won’t ever help
    Retryable, // retry may help
    Unknown,
}

pub enum Effect {
    None,    // definitely no side effects
    Some,    // definitely did something
    Unknown, // maybe / don’t know
}
```

We *only* add these where they drive real behavior (HTTP layer, job runner, etc.), and we let them live on canonical error enums:

```rust
impl StorageError {
    pub fn blame(&self) -> Blame { /* match self */ }
    pub fn transience(&self) -> Transience { /* match self */ }
    pub fn effect(&self) -> Effect { /* match self */ }
}
```

Rules of thumb:

* Don’t feel obligated to attach full geometry to every leaf type.
* Start with a single `ErrorClass` if that’s enough.
* Avoid `_ =>` in matches so adding a new variant forces you to decide how to classify it.

---

## 7. Smells and refactors

Quick things to watch for in reviews:

### 7.1 Global god‑error

```rust
pub enum Error {
    Io(std::io::Error),
    Http(reqwest::Error),
    Db(sqlx::Error),
    // ...
}
```

Refactor:

* Split by capability: `StorageError`, `UserRepoError`, `MailerError`, `PaymentError`.
* Map those into a boundary type (`ApiError`, `JobError`) only at the edge.

### 7.2 Variants named after crates

```rust
pub enum SendError {
    Reqwest(reqwest::Error),
    Smtp(smtp::Error),
}
```

Refactor:

```rust
pub enum SendError {
    Transport { endpoint: Url, #[source] source: reqwest::Error },
    Protocol  { detail: String, #[source] source: smtp::Error },
}
```

### 7.3 `Other(String)` / `Unknown` catch‑alls

Use sparingly and treat them like `unwrap`: allowed, but obvious debt.

* For each place that constructs `Other`, ask: should this be a real variant?
* If everything drifts into `Other`, you’re not modelling your domain.

### 7.4 Logging at every layer

Refactor toward:

* inner layers: no logging, just structured errors,
* outer layer: one log entry with full context + error chain.

### 7.5 Panics for world‑caused failures

Refactor:

* “File missing”, “invalid JSON”, “host unreachable” etc. → proper error types.
* Panic only on internal invariants and impossible states.

---

## 8. Checklists

### 8.1 When you add a new `Result<T, E>`

Ask:

1. **What capability is this part of?**
   Is there a canonical error enum already (e.g. `StorageError`)? Should there be?

2. **Who is the first real caller of this error?**
   HTTP handler, job, CLI, other component?

3. **What distinct behaviors do they need?**
   Retry? Ask user to change input? Show “not found”? Crash?

4. **Does each distinct behavior have a variant?**
   If not, split or merge variants.

5. **Can I express some preconditions in types instead of errors?**

6. **Am I leaking dependency details in the public surface?**
   If yes, move those into `#[source]`.

7. **Do I need internal error sets here, or is the canonical enum enough?**
   Only reach for `OneOf` if it simplifies real composition.

### 8.2 Reviewing an error‑heavy PR

Skim for:

* Global “god” error enums.
* Enums named after libraries, not domain behaviors.
* `Other(String)` doing too much work.
* Logging from deep internals instead of boundaries.
* Panics/`unwrap` reachable via user input or environment.
* `Result<T, anyhow::Error>` in capability traits (fine at very outer edges, not fine inside core).

Push toward:

* leaf error types where useful,
* one canonical error enum per real capability,
* internal error sets (`OneOf`) only when they simplify composition,
* classification only where it feeds real policy or observability.

---

## 9. What we actually want errors to be

* Types state what’s allowed.
* Traits carve capabilities at real joints.
* Errors admit where that story stops matching reality.

If we get them right:

* Behavior at boundaries (HTTP codes, retries, alerts) falls out of the types instead of being folklore.
* We stop saying “I/O error” and start saying “auth service at `auth‑01` is down”.
* “We don’t know if we charged the card” becomes a named variant instead of a cursed log line.

We don’t need a grand unified error framework.

We need:

* one honest capability error enum per capability,
* leaf error types where they make the story clearer,
* `OneOf`‑style sets internally when they actually help compose things,
* and, where it pays off, a small derived classification to drive policy and observability.

Design errors with the same care you put into types and traits.

They’re not leftovers from the happy path.
They’re the other half of the map.
