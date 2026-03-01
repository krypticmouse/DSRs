use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokio::runtime::{Handle, RuntimeFlavor};

use crate::LM;
use crate::core::lm::{Chat, Message, ToolLoopMode};

#[async_trait]
pub trait LlmQuery: Send + Sync {
    async fn query(&self, prompt: &str) -> anyhow::Result<String>;
}

#[async_trait]
impl LlmQuery for LM {
    async fn query(&self, prompt: &str) -> anyhow::Result<String> {
        let mut msgs = Vec::new();
        if let Some(sys) = &self.system_prompt {
            msgs.push(Message::system(sys));
        }
        msgs.push(Message::user(prompt));
        let messages = Chat::new(msgs);
        let response = self
            .call(messages, Vec::new(), ToolLoopMode::CallerManaged)
            .await?;
        Ok(response.output.text_content())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct LlmTools {
    lm: Arc<dyn LlmQuery>,
    pub max_llm_calls: usize,
    budget_remaining: Arc<AtomicUsize>,
    handle: Handle,
}

impl LlmTools {
    pub fn new(
        lm: Arc<dyn LlmQuery>,
        budget_remaining: Arc<AtomicUsize>,
        max_llm_calls: usize,
        handle: Handle,
    ) -> Self {
        Self {
            lm,
            max_llm_calls,
            budget_remaining,
            handle,
        }
    }

    pub fn with_budget(lm: Arc<dyn LlmQuery>, max_llm_calls: usize, handle: Handle) -> Self {
        Self::new(
            lm,
            Arc::new(AtomicUsize::new(max_llm_calls)),
            max_llm_calls,
            handle,
        )
    }

    #[cfg(test)]
    fn call_count(&self) -> usize {
        self.max_llm_calls
            .saturating_sub(self.budget_remaining.load(Ordering::SeqCst))
    }

    pub fn remaining_calls(&self) -> usize {
        self.budget_remaining.load(Ordering::SeqCst)
    }

    fn reserve_calls(&self, count: usize) -> PyResult<()> {
        loop {
            let current = self.budget_remaining.load(Ordering::SeqCst);
            if current < count {
                return Err(PyRuntimeError::new_err(format!(
                    "[Error] RuntimeError: LLM call budget exhausted: requested {count}, remaining {current}, max {}. This is retryable after reducing llm_query usage.",
                    self.max_llm_calls
                )));
            }

            if self
                .budget_remaining
                .compare_exchange(current, current - count, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn reserve_calls_for_batch(&self, requested: usize) -> usize {
        loop {
            let current = self.budget_remaining.load(Ordering::SeqCst);
            let to_execute = current.min(requested);
            if self
                .budget_remaining
                .compare_exchange(
                    current,
                    current.saturating_sub(to_execute),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return to_execute;
            }
        }
    }

    fn emit_budget_warning(&self, executed: usize, requested: usize) {
        if executed >= requested {
            return;
        }
        let remaining = self.remaining_calls();
        let warning = format!(
            "⚠ Budget: executed first {executed} of {requested} requested queries ({remaining} remaining of {} max). \
results[i] aligns to prompts[i] for i < {executed}; skipped prompts[{executed}..{requested}].",
            self.max_llm_calls,
        );
        Python::attach(|py| {
            if let Ok(builtins) = PyModule::import(py, "builtins")
                && let Ok(print_fn) = builtins.getattr("print")
            {
                let _ = print_fn.call1((warning,));
            }
        });
    }

    fn ensure_prompt(prompt: &str) -> PyResult<()> {
        if prompt.trim().is_empty() {
            return Err(PyValueError::new_err(
                "[Error] ValueError: prompt cannot be empty",
            ));
        }
        Ok(())
    }

    fn block_with_runtime<F, T>(&self, fut: F) -> PyResult<T>
    where
        F: Future<Output = T>,
    {
        let current_handle = Handle::try_current().map_err(|err| {
            Self::runtime_error(format!("an active Tokio runtime is required: {err}"))
        })?;
        if current_handle.runtime_flavor() == RuntimeFlavor::CurrentThread {
            return Err(Self::runtime_error(
                "llm_query requires a multi-thread Tokio runtime; current-thread runtime is not supported",
            ));
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::task::block_in_place(|| self.handle.block_on(fut))
        }))
        .map_err(|_| {
            Self::runtime_error(
                "failed to block in the current Tokio runtime; use a multi-thread runtime",
            )
        })
    }

    fn runtime_error(err: impl std::fmt::Display) -> PyErr {
        PyRuntimeError::new_err(format!("[Error] RuntimeError: {err}"))
    }
}

#[pymethods]
impl LlmTools {
    fn llm_query(&self, prompt: String) -> PyResult<String> {
        Self::ensure_prompt(&prompt)?;
        self.reserve_calls(1)?;

        let response = self.block_with_runtime(self.lm.query(&prompt))?;
        let response = response.map_err(Self::runtime_error)?;

        Ok(response)
    }

    fn llm_query_batched(&self, prompts: Vec<String>) -> PyResult<Vec<String>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        for prompt in &prompts {
            Self::ensure_prompt(prompt)?;
        }

        let requested = prompts.len();
        let executable = self.reserve_calls_for_batch(requested);
        if executable == 0 {
            self.emit_budget_warning(0, requested);
            return Ok(Vec::new());
        }
        self.emit_budget_warning(executable, requested);

        let responses = self.block_with_runtime(async {
            let futures = prompts
                .iter()
                .take(executable)
                .map(|prompt| self.lm.query(prompt));
            futures::future::join_all(futures).await
        })?;

        let mut results = Vec::with_capacity(responses.len());
        for response in responses {
            match response {
                Ok(text) => results.push(text),
                Err(err) => return Err(Self::runtime_error(err)),
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct MockLm {
        calls: Mutex<Vec<String>>,
        fail_on: Mutex<HashSet<String>>,
    }

    #[async_trait]
    impl LlmQuery for MockLm {
        async fn query(&self, prompt: &str) -> anyhow::Result<String> {
            self.calls
                .lock()
                .expect("calls mutex poisoned")
                .push(prompt.to_string());

            if self
                .fail_on
                .lock()
                .expect("fail_on mutex poisoned")
                .contains(prompt)
            {
                anyhow::bail!("mock failure for {prompt}");
            }

            Ok(format!("answer:{prompt}"))
        }
    }

    #[test]
    fn llm_query_consumes_budget_and_returns_text() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let lm = Arc::new(MockLm::default());
            let tools = LlmTools::with_budget(lm.clone(), 2, Handle::current());

            let first = tools.llm_query("hello".to_string()).expect("first call");
            assert_eq!(first, "answer:hello");
            assert_eq!(tools.call_count(), 1);

            let second = tools.llm_query("world".to_string()).expect("second call");
            assert_eq!(second, "answer:world");
            assert_eq!(tools.call_count(), 2);

            let calls = lm.calls.lock().expect("calls lock").clone();
            assert_eq!(calls, vec!["hello".to_string(), "world".to_string()]);
        });
    }

    #[test]
    fn budget_exhaustion_returns_retryable_runtime_error() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let tools = LlmTools::with_budget(Arc::new(MockLm::default()), 1, Handle::current());
            let _ = tools.llm_query("one".to_string()).expect("first call");

            let err = tools
                .llm_query("two".to_string())
                .expect_err("budget should be exhausted");
            assert!(err.to_string().contains("budget exhausted"));
            assert!(err.to_string().contains("retryable"));
        });
    }

    #[test]
    fn llm_query_batched_runs_all_prompts() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let lm = Arc::new(MockLm::default());
            let tools = LlmTools::with_budget(lm.clone(), 5, Handle::current());

            let responses = tools
                .llm_query_batched(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .expect("batched call");
            assert_eq!(responses, vec!["answer:a", "answer:b", "answer:c"]);
            assert_eq!(tools.call_count(), 3);

            let mut calls = lm.calls.lock().expect("calls lock").clone();
            calls.sort();
            assert_eq!(calls, vec!["a", "b", "c"]);
        });
    }

    #[test]
    fn llm_query_batched_propagates_runtime_errors() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let lm = Arc::new(MockLm::default());
            lm.fail_on
                .lock()
                .expect("fail_on lock")
                .insert("bad".to_string());

            let tools = LlmTools::with_budget(lm, 3, Handle::current());
            let err = tools
                .llm_query_batched(vec!["ok".to_string(), "bad".to_string()])
                .expect_err("second prompt should fail");

            assert!(err.to_string().contains("mock failure for bad"));
        });
    }

    #[test]
    fn llm_query_batched_executes_partial_batch_when_budget_is_short() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let lm = Arc::new(MockLm::default());
            let tools = LlmTools::with_budget(lm.clone(), 2, Handle::current());

            let responses = tools
                .llm_query_batched(vec![
                    "one".to_string(),
                    "two".to_string(),
                    "three".to_string(),
                ])
                .expect("partial batch should succeed");
            assert_eq!(responses, vec!["answer:one", "answer:two"]);
            assert_eq!(tools.remaining_calls(), 0);

            let calls = lm.calls.lock().expect("calls lock").clone();
            assert_eq!(calls, vec!["one".to_string(), "two".to_string()]);
        });
    }

    #[test]
    fn llm_query_batched_returns_empty_when_budget_is_zero() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let lm = Arc::new(MockLm::default());
            let tools = LlmTools::with_budget(lm.clone(), 1, Handle::current());
            let _ = tools.llm_query("one".to_string()).expect("first call");
            assert_eq!(tools.remaining_calls(), 0);

            let responses = tools
                .llm_query_batched(vec!["two".to_string(), "three".to_string()])
                .expect("zero-budget batch should not error");
            assert!(responses.is_empty());

            let calls = lm.calls.lock().expect("calls lock").clone();
            assert_eq!(calls, vec!["one".to_string()]);
        });
    }

    #[test]
    fn shared_budget_is_enforced_across_single_and_batched_calls() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let tools = LlmTools::with_budget(Arc::new(MockLm::default()), 3, Handle::current());

            let first = tools.llm_query("one".to_string()).expect("first call");
            assert_eq!(first, "answer:one");
            assert_eq!(tools.remaining_calls(), 2);

            let responses = tools
                .llm_query_batched(vec!["two".to_string(), "three".to_string()])
                .expect("batched call");
            assert_eq!(responses, vec!["answer:two", "answer:three"]);
            assert_eq!(tools.remaining_calls(), 0);

            let err = tools
                .llm_query("four".to_string())
                .expect_err("budget should be exhausted");
            assert!(err.to_string().contains("budget exhausted"));
        });
    }

    #[test]
    fn empty_batched_call_returns_immediately_without_consuming_budget() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let tools = LlmTools::with_budget(Arc::new(MockLm::default()), 2, Handle::current());

            let responses = tools
                .llm_query_batched(Vec::new())
                .expect("empty batch should be valid");
            assert!(responses.is_empty());
            assert_eq!(tools.remaining_calls(), 2);
        });
    }

    #[test]
    fn current_thread_runtime_returns_clear_error_instead_of_panicking() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");

        rt.block_on(async {
            let tools = LlmTools::with_budget(Arc::new(MockLm::default()), 1, Handle::current());
            let err = tools
                .llm_query("hello".to_string())
                .expect_err("current-thread runtime should fail gracefully");

            let message = err.to_string();
            assert!(message.contains("multi-thread Tokio runtime"));
            assert!(message.contains("current-thread"));
        });
    }
}
