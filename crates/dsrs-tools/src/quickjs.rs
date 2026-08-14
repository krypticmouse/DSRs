//! Tier-1 executor: in-process QuickJS (quickjs-ng via `rquickjs`).
//!
//! Every tool call gets a **fresh** runtime + context (creation is on the
//! order of a hundred microseconds, so isolation is cheaper than sharing),
//! with:
//!
//! - a per-call memory limit ([`Runtime::set_memory_limit`]),
//! - a wall-clock deadline enforced by the engine's interrupt handler (a
//!   runaway `while(true)` is killed and reported as [`ExecError::Timeout`]),
//! - no ambient authority: no filesystem, no network, no environment, no
//!   module loader — the only host access is through explicitly injected
//!   [`Capability`] functions,
//! - bytecode reuse: sources are compiled once per unique content
//!   (BLAKE3-keyed) and the bytecode is shared across calls.
//!
//! The blocking QuickJS work runs on Tokio's blocking pool; injected
//! capabilities are async Rust and are driven to completion from the sandbox
//! thread via `Handle::block_on`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use rquickjs::function::Func;
use rquickjs::{
    CatchResultExt, CaughtError, Context, Ctx, Error as JsError, Exception, Function, Module,
    Runtime, Value as JsValue, WriteOptions,
};
use serde_json::Value;
use tokio::runtime::Handle;

use crate::capability::Capability;
use crate::error::{ExecError, RegisterError};
use crate::executor::{Executor, RegisteredTool, ToolInvocation};
use crate::source::ToolSource;

/// Marker embedded in exceptions thrown by failed capability calls so the
/// classifier can attribute the failure to the capability rather than the JS.
const CAP_ERROR_MARKER: &str = "__dsrs_capability_error__";
/// Module name used for every compiled tool (fixed so bytecode depends only on
/// source content and stays cacheable across tool names).
const MODULE_NAME: &str = "dsrs_tool";

/// Resource limits applied to every sandbox instance.
#[derive(Debug, Clone, Copy)]
pub struct SandboxConfig {
    /// Max heap for one call, in bytes. Default: 32 MiB.
    pub memory_limit: usize,
    /// Wall-clock budget for one call. Time spent inside a capability counts
    /// against the budget but cannot be interrupted mid-call. Default: 500ms.
    pub deadline: Duration,
    /// Max JS stack, in bytes. Default: 512 KiB.
    pub max_stack: usize,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit: 32 * 1024 * 1024,
            deadline: Duration::from_millis(500),
            max_stack: 512 * 1024,
        }
    }
}

#[derive(Clone)]
struct ToolEntry {
    meta: RegisteredTool,
    bytecode: Arc<Vec<u8>>,
    required: Arc<Vec<String>>,
}

#[derive(Default)]
struct BytecodeCache {
    by_hash: Mutex<HashMap<[u8; 32], Arc<Vec<u8>>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Counters for the content-hash bytecode cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
}

/// The Tier-1 in-process QuickJS executor. See the module docs.
///
/// Cheap to share: wrap it in an [`Arc`] and clone across tasks.
pub struct QuickJsExecutor {
    config: SandboxConfig,
    capabilities: RwLock<Vec<Capability>>,
    tools: RwLock<HashMap<String, ToolEntry>>,
    cache: BytecodeCache,
}

impl Default for QuickJsExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for QuickJsExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuickJsExecutor")
            .field("config", &self.config)
            .field("capabilities", &self.capability_names())
            .field(
                "tools",
                &self
                    .tools
                    .read()
                    .expect("tool lock poisoned")
                    .keys()
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl QuickJsExecutor {
    /// Executor with default [`SandboxConfig`] and no capabilities.
    pub fn new() -> Self {
        Self::with_config(SandboxConfig::default())
    }

    pub fn with_config(config: SandboxConfig) -> Self {
        Self {
            config,
            capabilities: RwLock::new(Vec::new()),
            tools: RwLock::new(HashMap::new()),
            cache: BytecodeCache::default(),
        }
    }

    pub fn builder() -> QuickJsExecutorBuilder {
        QuickJsExecutorBuilder::default()
    }

    pub fn config(&self) -> SandboxConfig {
        self.config
    }

    /// Inject a host capability. Its name becomes a global JS function inside
    /// every sandbox created from now on.
    pub fn add_capability(&self, capability: Capability) -> Result<(), RegisterError> {
        Capability::validate_name(capability.name())?;
        let mut caps = self.capabilities.write().expect("capability lock poisoned");
        if caps.iter().any(|c| c.name() == capability.name()) {
            return Err(RegisterError::InvalidCapability {
                name: capability.name().to_string(),
                reason: "a capability with this name is already registered".to_string(),
            });
        }
        caps.push(capability);
        Ok(())
    }

    /// Names of the injected capabilities, in registration order.
    pub fn capability_names(&self) -> Vec<String> {
        self.capabilities
            .read()
            .expect("capability lock poisoned")
            .iter()
            .map(|c| c.name().to_string())
            .collect()
    }

    /// Bytecode-cache counters (entries, hits, misses).
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            entries: self
                .cache
                .by_hash
                .lock()
                .expect("cache lock poisoned")
                .len(),
            hits: self.cache.hits.load(Ordering::Relaxed),
            misses: self.cache.misses.load(Ordering::Relaxed),
        }
    }

    /// Wrap a registered tool as a [`rig::tool::ToolDyn`] so it can be passed
    /// anywhere DSRs already accepts tools.
    pub fn rig_tool(
        self: &Arc<Self>,
        name: &str,
    ) -> Option<Arc<dyn rig::tool::ToolDyn + Send + Sync>> {
        let meta = self.tool(name)?;
        Some(Arc::new(crate::rig_tool::SandboxTool::new(
            Arc::clone(self) as Arc<dyn Executor>,
            meta,
        )))
    }

    /// [`register`](Executor::register) + [`rig_tool`](Self::rig_tool) in one
    /// step: the full LATM lifecycle ending in a rig-compatible tool.
    pub async fn register_rig(
        self: &Arc<Self>,
        source: ToolSource,
    ) -> Result<Arc<dyn rig::tool::ToolDyn + Send + Sync>, RegisterError> {
        let meta = self.register(source).await?;
        Ok(Arc::new(crate::rig_tool::SandboxTool::new(
            Arc::clone(self) as Arc<dyn Executor>,
            meta,
        )))
    }

    /// Synchronous execution on the **current** thread, skipping the Tokio
    /// blocking pool. Useful from non-async code and for measuring raw
    /// sandbox latency. Capabilities still need a reachable Tokio runtime
    /// ([`Handle::try_current`]); without one, capability calls fail.
    ///
    /// Do not call this from inside an async task: it blocks the thread for
    /// the whole sandbox run.
    pub fn execute_blocking(&self, invocation: ToolInvocation) -> Result<Value, ExecError> {
        let (name, entry, args_json) = self.prepare(invocation)?;
        run_tool(
            &name,
            &entry.bytecode,
            &args_json,
            &self.snapshot_capabilities(),
            self.config,
            Handle::try_current().ok(),
        )
    }

    /// Lookup + argument validation shared by the sync and async paths.
    fn prepare(
        &self,
        invocation: ToolInvocation,
    ) -> Result<(String, ToolEntry, String), ExecError> {
        let ToolInvocation { name, args } = invocation;
        let entry = self
            .tools
            .read()
            .expect("tool lock poisoned")
            .get(&name)
            .cloned()
            .ok_or_else(|| ExecError::NotFound { name: name.clone() })?;
        validate_args(&name, &args, &entry.required)?;
        let args_json = args.to_string();
        Ok((name, entry, args_json))
    }

    fn snapshot_capabilities(&self) -> Vec<Capability> {
        self.capabilities
            .read()
            .expect("capability lock poisoned")
            .clone()
    }

    /// Compile `js_source` (wrapped as an ES module) to bytecode, reusing the
    /// content-hash cache. Returns `(bytecode, hex_hash, was_cache_hit)`.
    fn compile_or_cached(
        &self,
        js_source: &str,
    ) -> Result<(Arc<Vec<u8>>, String, bool), RegisterError> {
        let hash = *blake3::hash(js_source.as_bytes()).as_bytes();
        let hex = blake3::Hash::from_bytes(hash).to_hex().to_string();

        if let Some(bytecode) = self
            .cache
            .by_hash
            .lock()
            .expect("cache lock poisoned")
            .get(&hash)
            .cloned()
        {
            self.cache.hits.fetch_add(1, Ordering::Relaxed);
            return Ok((bytecode, hex, true));
        }

        let bytecode = Arc::new(compile_module(&wrap_source(js_source), self.config)?);
        self.cache
            .by_hash
            .lock()
            .expect("cache lock poisoned")
            .insert(hash, Arc::clone(&bytecode));
        self.cache.misses.fetch_add(1, Ordering::Relaxed);
        Ok((bytecode, hex, false))
    }
}

/// Builder for [`QuickJsExecutor`].
#[derive(Default)]
pub struct QuickJsExecutorBuilder {
    config: SandboxConfig,
    capabilities: Vec<Capability>,
}

impl QuickJsExecutorBuilder {
    /// Max heap per call, in bytes.
    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.config.memory_limit = bytes;
        self
    }

    /// Wall-clock budget per call.
    pub fn deadline(mut self, deadline: Duration) -> Self {
        self.config.deadline = deadline;
        self
    }

    /// Max JS stack per call, in bytes.
    pub fn max_stack(mut self, bytes: usize) -> Self {
        self.config.max_stack = bytes;
        self
    }

    /// Inject a host capability (validated at [`build`](Self::build)).
    pub fn capability(mut self, capability: Capability) -> Self {
        self.capabilities.push(capability);
        self
    }

    pub fn build(self) -> Result<QuickJsExecutor, RegisterError> {
        let executor = QuickJsExecutor::with_config(self.config);
        for capability in self.capabilities {
            executor.add_capability(capability)?;
        }
        Ok(executor)
    }
}

#[async_trait]
impl Executor for QuickJsExecutor {
    fn validate(&self, source: &ToolSource) -> Result<(), RegisterError> {
        source.validate_shape()?;
        if self
            .tools
            .read()
            .expect("tool lock poisoned")
            .contains_key(&source.name)
        {
            return Err(RegisterError::Duplicate {
                name: source.name.clone(),
            });
        }
        Ok(())
    }

    async fn register(&self, source: ToolSource) -> Result<RegisteredTool, RegisterError> {
        // Stage 1: structural validation (name, schema shape, duplicates).
        self.validate(&source)?;

        // Stage 2+3: compile and self-test on the blocking pool.
        let handle = current_handle().map_err(RegisterError::Execution)?;
        let config = self.config;
        let capabilities = self.snapshot_capabilities();

        // Stage 2: parse/compile (content-hash cached).
        let (bytecode, source_hash, _hit) = self.compile_or_cached(&source.js_source)?;

        // Stage 3: instantiate in a sandbox, check the module evaluates to a
        // function, and run the self-test if present.
        let name = source.name.clone();
        let self_test = source.self_test.clone();
        let self_tested = self_test.is_some();
        let validation_bytecode = Arc::clone(&bytecode);
        let validation_handle = handle.clone();
        handle
            .spawn_blocking(move || {
                run_validation(
                    &name,
                    &validation_bytecode,
                    self_test.as_deref(),
                    &capabilities,
                    config,
                    validation_handle,
                )
            })
            .await
            .map_err(|e| {
                RegisterError::Execution(ExecError::Internal {
                    message: format!("sandbox validation task failed: {e}"),
                })
            })??;

        // Stage 4: register.
        let meta = RegisteredTool {
            name: source.name.clone(),
            description: source.description.clone(),
            parameters: source.params.clone(),
            source_hash,
            self_tested,
        };
        let entry = ToolEntry {
            meta: meta.clone(),
            bytecode,
            required: Arc::new(source.required_params()),
        };
        let mut tools = self.tools.write().expect("tool lock poisoned");
        if tools.contains_key(&source.name) {
            return Err(RegisterError::Duplicate { name: source.name });
        }
        tools.insert(source.name, entry);
        Ok(meta)
    }

    async fn execute(&self, invocation: ToolInvocation) -> Result<Value, ExecError> {
        let (name, entry, args_json) = self.prepare(invocation)?;

        let handle = current_handle()?;
        let config = self.config;
        let capabilities = self.snapshot_capabilities();
        let bytecode = Arc::clone(&entry.bytecode);
        let cap_handle = handle.clone();
        handle
            .spawn_blocking(move || {
                run_tool(
                    &name,
                    &bytecode,
                    &args_json,
                    &capabilities,
                    config,
                    Some(cap_handle),
                )
            })
            .await
            .map_err(|e| ExecError::Internal {
                message: format!("sandbox task failed: {e}"),
            })?
    }

    fn tool(&self, name: &str) -> Option<RegisteredTool> {
        self.tools
            .read()
            .expect("tool lock poisoned")
            .get(name)
            .map(|entry| entry.meta.clone())
    }

    fn tools(&self) -> Vec<RegisteredTool> {
        self.tools
            .read()
            .expect("tool lock poisoned")
            .values()
            .map(|entry| entry.meta.clone())
            .collect()
    }

    fn deregister(&self, name: &str) -> bool {
        self.tools
            .write()
            .expect("tool lock poisoned")
            .remove(name)
            .is_some()
    }
}

/// Execute a standalone JavaScript program in a fresh, fully fenced sandbox
/// with `capabilities` injected — the Code Mode execution primitive.
///
/// The source runs as the body of an async IIFE: top-level `return` produces
/// the result (a JSON-serializable value; `undefined` maps to `null`), and
/// `await` is tolerated but only microtask-resolvable promises settle (there
/// is no event loop). Capabilities appear as plain global functions. Errors
/// are attributed to the pseudo-tool name
/// [`RUN_JS_TOOL_NAME`](crate::RUN_JS_TOOL_NAME); a syntax error surfaces as
/// [`ExecError::Js`] so a generating LLM can repair the script.
pub async fn run_script(
    source: &str,
    capabilities: Vec<Capability>,
    config: SandboxConfig,
) -> Result<Value, ExecError> {
    let handle = current_handle()?;
    let source = source.to_string();
    let cap_handle = handle.clone();
    handle
        .spawn_blocking(move || run_script_blocking(&source, &capabilities, config, Some(cap_handle)))
        .await
        .map_err(|e| ExecError::Internal {
            message: format!("sandbox task failed: {e}"),
        })?
}

fn run_script_blocking(
    source: &str,
    capabilities: &[Capability],
    config: SandboxConfig,
    handle: Option<Handle>,
) -> Result<Value, ExecError> {
    let name = crate::code_mode::RUN_JS_TOOL_NAME;
    let sandbox = Sandbox::create(config, capabilities, handle)?;
    let wrapped = format!("export default (async () => {{\n{source}\n}})();");
    sandbox.context.with(|ctx| {
        let module = Module::declare(ctx.clone(), "dsrs_script", wrapped)
            .catch(&ctx)
            .map_err(|caught| classify(name, &caught, &sandbox))?;
        let (module, promise) = module
            .eval()
            .catch(&ctx)
            .map_err(|caught| classify(name, &caught, &sandbox))?;
        promise
            .finish::<()>()
            .catch(&ctx)
            .map_err(|caught| classify(name, &caught, &sandbox))?;
        let export: JsValue<'_> = module
            .namespace()
            .and_then(|ns| ns.get("default"))
            .catch(&ctx)
            .map_err(|caught| classify(name, &caught, &sandbox))?;
        // The async IIFE returns a promise; settle it on the microtask queue.
        let result = match export.as_promise() {
            Some(promise) => promise
                .finish::<JsValue<'_>>()
                .catch(&ctx)
                .map_err(|caught| classify(name, &caught, &sandbox))?,
            None => export,
        };
        let serialized = ctx
            .json_stringify(result)
            .catch(&ctx)
            .map_err(|caught| classify(name, &caught, &sandbox))?;
        match serialized {
            None => Ok(Value::Null),
            Some(json) => {
                let json = json.to_string().map_err(|e| ExecError::Internal {
                    message: format!("result was not valid UTF-8: {e}"),
                })?;
                serde_json::from_str(&json).map_err(|e| ExecError::Internal {
                    message: format!("result round-trip failed: {e}"),
                })
            }
        }
    })
}

fn current_handle() -> Result<Handle, ExecError> {
    Handle::try_current().map_err(|_| ExecError::Internal {
        message: "QuickJsExecutor requires a Tokio runtime (used for its blocking pool)"
            .to_string(),
    })
}

/// Wrap the user source as the default export of an ES module. The source
/// contract (see [`ToolSource`]) is "an expression evaluating to a function";
/// parenthesizing also turns named `function` declarations into expressions.
fn wrap_source(js_source: &str) -> String {
    let mut trimmed = js_source.trim();
    while let Some(stripped) = trimmed.strip_suffix(';') {
        trimmed = stripped.trim_end();
    }
    format!("export default (\n{trimmed}\n);")
}

fn validate_args(name: &str, args: &Value, required: &[String]) -> Result<(), ExecError> {
    let Some(map) = args.as_object() else {
        return Err(ExecError::InvalidArgs {
            name: name.to_string(),
            reason: format!(
                "arguments must be a JSON object, got {}",
                crate::source::json_type_name(args)
            ),
        });
    };
    let missing: Vec<&String> = required.iter().filter(|k| !map.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(ExecError::InvalidArgs {
            name: name.to_string(),
            reason: format!(
                "missing required argument(s): {}",
                missing
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    Ok(())
}

/// Compile a wrapped module source to QuickJS bytecode in a throwaway,
/// resource-limited runtime. Parse errors come back as `Compile`.
fn compile_module(wrapped_source: &str, config: SandboxConfig) -> Result<Vec<u8>, RegisterError> {
    let sandbox = Sandbox::create(config, &[], None).map_err(RegisterError::Execution)?;
    sandbox.context.with(|ctx| {
        let module = Module::declare(ctx.clone(), MODULE_NAME, wrapped_source)
            .catch(&ctx)
            .map_err(|caught| RegisterError::Compile {
                message: caught.to_string(),
            })?;
        module
            .write(WriteOptions::default())
            .map_err(|e| RegisterError::Compile {
                message: format!("failed to serialize bytecode: {e}"),
            })
    })
}

/// One fresh, fully fenced QuickJS instance: limits armed, deadline
/// interrupt installed, capabilities injected. Dropped after a single use.
struct Sandbox {
    // Field order matters: `context` must drop before `runtime`.
    context: Context,
    #[allow(dead_code)]
    runtime: Runtime,
    timed_out: Arc<AtomicBool>,
    config: SandboxConfig,
}

impl Sandbox {
    fn create(
        config: SandboxConfig,
        capabilities: &[Capability],
        handle: Option<Handle>,
    ) -> Result<Self, ExecError> {
        let internal = |message: String| ExecError::Internal { message };

        let runtime =
            Runtime::new().map_err(|e| internal(format!("failed to create runtime: {e}")))?;
        runtime.set_memory_limit(config.memory_limit);
        runtime.set_max_stack_size(config.max_stack);

        let timed_out = Arc::new(AtomicBool::new(false));
        let deadline = Instant::now() + config.deadline;
        let flag = Arc::clone(&timed_out);
        runtime.set_interrupt_handler(Some(Box::new(move || {
            if Instant::now() >= deadline {
                flag.store(true, Ordering::SeqCst);
                true
            } else {
                false
            }
        })));

        let context = Context::full(&runtime)
            .map_err(|e| internal(format!("failed to create context: {e}")))?;

        if !capabilities.is_empty() {
            let handle = handle.ok_or_else(|| {
                internal("capabilities require a Tokio runtime handle".to_string())
            })?;
            context.with(|ctx| install_capabilities(&ctx, capabilities, &handle))?;
        }

        Ok(Self {
            context,
            runtime,
            timed_out,
            config,
        })
    }
}

/// Expose each capability as a global JS function. The raw hook takes and
/// returns JSON strings; a JS shim gives callers a plain-value API.
fn install_capabilities(
    ctx: &Ctx<'_>,
    capabilities: &[Capability],
    handle: &Handle,
) -> Result<(), ExecError> {
    let globals = ctx.globals();
    for capability in capabilities {
        let cap_name = capability.name().to_string();
        let hook_name = format!("__dsrs_cap_{cap_name}");
        let handler = capability.handler();
        let handle = handle.clone();
        let thrown_name = cap_name.clone();

        let hook = move |ctx: Ctx<'_>, args_json: String| -> rquickjs::Result<String> {
            let throw = |message: String| {
                Exception::throw_message(
                    &ctx,
                    &format!("{CAP_ERROR_MARKER}:{thrown_name}:{message}"),
                )
            };
            let args: Value = serde_json::from_str(&args_json)
                .map_err(|e| throw(format!("argument round-trip failed: {e}")))?;
            let result = handle.block_on(handler(args)).map_err(throw)?;
            serde_json::to_string(&result)
                .map_err(|e| throw(format!("result serialization failed: {e}")))
        };

        globals
            .set(hook_name.as_str(), Func::from(hook))
            .map_err(|e| ExecError::Internal {
                message: format!("failed to install capability `{cap_name}`: {e}"),
            })?;

        // Shim: plain JS values in/out; `undefined` argument becomes `null`.
        let shim = format!(
            "globalThis.{cap_name} = ((raw) => (arg) => JSON.parse(raw(JSON.stringify(arg ?? null))))(globalThis.{hook_name}); delete globalThis.{hook_name};"
        );
        ctx.eval::<(), _>(shim.into_bytes())
            .map_err(|e| ExecError::Internal {
                message: format!("failed to install capability shim `{cap_name}`: {e}"),
            })?;
    }
    Ok(())
}

/// Load cached bytecode, evaluate the module, and return its default export
/// (the tool function).
fn instantiate_tool<'js>(
    ctx: &Ctx<'js>,
    bytecode: &[u8],
) -> Result<Function<'js>, ToolFailure<'js>> {
    // SAFETY: the bytecode was produced by `Module::write` from this same
    // quickjs-ng build (the in-memory cache never persists across builds).
    let module = unsafe { Module::load(ctx.clone(), bytecode) }
        .catch(ctx)
        .map_err(ToolFailure::Caught)?;
    let (module, promise) = module.eval().catch(ctx).map_err(ToolFailure::Caught)?;
    promise
        .finish::<()>()
        .catch(ctx)
        .map_err(ToolFailure::Caught)?;
    let export: JsValue<'js> = module
        .namespace()
        .and_then(|ns| ns.get("default"))
        .catch(ctx)
        .map_err(ToolFailure::Caught)?;
    let type_name = js_typeof_name(export.type_name());
    export
        .into_function()
        .ok_or(ToolFailure::NotAFunction { type_name })
}

enum ToolFailure<'js> {
    Caught(CaughtError<'js>),
    NotAFunction { type_name: &'static str },
}

/// Map rquickjs' internal type names onto JavaScript `typeof` vocabulary so
/// errors speak the language the generating LLM wrote.
fn js_typeof_name(rquickjs_name: &'static str) -> &'static str {
    match rquickjs_name {
        "int" | "float" => "number",
        "bool" => "boolean",
        "uninitialized" => "undefined",
        other => other,
    }
}

fn run_validation(
    tool_name: &str,
    bytecode: &[u8],
    self_test: Option<&str>,
    capabilities: &[Capability],
    config: SandboxConfig,
    handle: Handle,
) -> Result<(), RegisterError> {
    let sandbox =
        Sandbox::create(config, capabilities, Some(handle)).map_err(RegisterError::Execution)?;

    sandbox.context.with(|ctx| {
        let function = match instantiate_tool(&ctx, bytecode) {
            Ok(function) => function,
            Err(ToolFailure::NotAFunction { type_name }) => {
                return Err(RegisterError::NotAFunction {
                    evaluated_type: type_name.to_string(),
                });
            }
            Err(ToolFailure::Caught(caught)) => {
                return Err(match classify(tool_name, &caught, &sandbox) {
                    err @ (ExecError::Timeout { .. }
                    | ExecError::MemoryExceeded { .. }
                    | ExecError::Internal { .. }) => RegisterError::Execution(err),
                    other => RegisterError::Compile {
                        message: other.to_string(),
                    },
                });
            }
        };
        let Some(self_test) = self_test else {
            return Ok(());
        };
        ctx.globals().set("tool", function).map_err(|e| {
            RegisterError::Execution(ExecError::Internal {
                message: format!("failed to bind self-test global: {e}"),
            })
        })?;
        let outcome: Result<JsValue<'_>, CaughtError<'_>> = ctx
            .eval::<JsValue<'_>, _>(self_test.as_bytes().to_vec())
            .catch(&ctx);
        let value = match outcome {
            Ok(value) => value,
            Err(caught) => {
                return Err(match classify(tool_name, &caught, &sandbox) {
                    err @ (ExecError::Timeout { .. }
                    | ExecError::MemoryExceeded { .. }
                    | ExecError::Internal { .. }) => RegisterError::Execution(err),
                    other => RegisterError::SelfTest {
                        message: other.to_string(),
                    },
                });
            }
        };
        // Settle promise-returning self-tests on the microtask queue.
        let value = match value.as_promise() {
            Some(promise) => match promise.finish::<JsValue<'_>>().catch(&ctx) {
                Ok(value) => value,
                Err(caught) => {
                    return Err(RegisterError::SelfTest {
                        message: classify(tool_name, &caught, &sandbox).to_string(),
                    });
                }
            },
            None => value,
        };
        if value.as_bool() == Some(false) {
            return Err(RegisterError::SelfTest {
                message: "self-test completed with `false`".to_string(),
            });
        }
        Ok(())
    })
}

fn run_tool(
    tool_name: &str,
    bytecode: &[u8],
    args_json: &str,
    capabilities: &[Capability],
    config: SandboxConfig,
    handle: Option<Handle>,
) -> Result<Value, ExecError> {
    let sandbox = Sandbox::create(config, capabilities, handle)?;
    sandbox.context.with(|ctx| {
        let function = match instantiate_tool(&ctx, bytecode) {
            Ok(function) => function,
            Err(ToolFailure::NotAFunction { type_name }) => {
                return Err(ExecError::Internal {
                    message: format!(
                        "registered tool no longer evaluates to a function (got `{type_name}`)"
                    ),
                });
            }
            Err(ToolFailure::Caught(caught)) => {
                return Err(classify(tool_name, &caught, &sandbox));
            }
        };

        let args = ctx
            .json_parse(args_json.as_bytes().to_vec())
            .catch(&ctx)
            .map_err(|caught| classify(tool_name, &caught, &sandbox))?;

        let result: JsValue<'_> = function
            .call((args,))
            .catch(&ctx)
            .map_err(|caught| classify(tool_name, &caught, &sandbox))?;

        // Async tools: settle on the microtask queue (no event loop, so a
        // promise depending on timers/IO reports `PendingPromise`).
        let result = match result.as_promise() {
            Some(promise) => promise
                .finish::<JsValue<'_>>()
                .catch(&ctx)
                .map_err(|caught| classify(tool_name, &caught, &sandbox))?,
            None => result,
        };

        let serialized = ctx
            .json_stringify(result)
            .catch(&ctx)
            .map_err(|caught| classify(tool_name, &caught, &sandbox))?;
        match serialized {
            None => Ok(Value::Null),
            Some(json) => {
                let json = json.to_string().map_err(|e| ExecError::Internal {
                    message: format!("result was not valid UTF-8: {e}"),
                })?;
                serde_json::from_str(&json).map_err(|e| ExecError::Internal {
                    message: format!("result round-trip failed: {e}"),
                })
            }
        }
    })
}

/// Turn a caught sandbox error into a typed [`ExecError`], consulting the
/// deadline flag (authoritative for timeouts) and known engine messages.
fn classify(tool_name: &str, caught: &CaughtError<'_>, sandbox: &Sandbox) -> ExecError {
    let name = tool_name.to_string();
    if sandbox.timed_out.load(Ordering::SeqCst) {
        return ExecError::Timeout {
            name,
            deadline_ms: sandbox.config.deadline.as_millis() as u64,
        };
    }
    let memory = |name: String| ExecError::MemoryExceeded {
        name,
        limit_bytes: sandbox.config.memory_limit,
    };
    match caught {
        CaughtError::Error(JsError::Allocation) => memory(name),
        CaughtError::Error(JsError::WouldBlock) => ExecError::PendingPromise { name },
        CaughtError::Error(e) => ExecError::Internal {
            message: e.to_string(),
        },
        CaughtError::Exception(_) | CaughtError::Value(_) => {
            let message = caught.to_string();
            if message.contains("out of memory") {
                return memory(name);
            }
            if let Some((capability, cap_message)) = parse_capability_error(&message) {
                return ExecError::Capability {
                    name,
                    capability,
                    message: cap_message,
                };
            }
            ExecError::Js { name, message }
        }
    }
}

/// Extract `(capability_name, message)` from an exception whose message
/// contains the [`CAP_ERROR_MARKER`].
fn parse_capability_error(message: &str) -> Option<(String, String)> {
    let start = message.find(CAP_ERROR_MARKER)?;
    let rest = &message[start + CAP_ERROR_MARKER.len()..];
    let rest = rest.strip_prefix(':')?;
    let (name, tail) = rest.split_once(':')?;
    // Cut trailing stack lines the engine may append after the message.
    let tail = tail.lines().next().unwrap_or(tail).trim();
    Some((name.to_string(), tail.to_string()))
}
