#![cfg(feature = "rlm")]

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::runtime::Handle;

use crate::LM;

#[pyclass]
#[derive(Clone)]
pub struct LlmTools {
    lm: Arc<LM>,
    max_llm_calls: usize,
    call_count: Arc<Mutex<usize>>,
    runtime: Handle,
}

impl LlmTools {
    pub fn new(lm: Arc<LM>, max_llm_calls: usize, runtime: Handle) -> Self {
        Self {
            lm,
            max_llm_calls,
            call_count: Arc::new(Mutex::new(0)),
            runtime,
        }
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    fn reserve_calls(&self, count: usize) -> PyResult<()> {
        let mut current = self.call_count.lock().unwrap();
        if *current + count > self.max_llm_calls {
            return Err(PyRuntimeError::new_err(format!(
                "LLM call limit exceeded: {} + {} > {}. Use Python code for aggregation instead of making more LLM calls.",
                *current, count, self.max_llm_calls
            )));
        }
        *current += count;
        Ok(())
    }

    fn ensure_prompt(prompt: &str) -> PyResult<()> {
        if prompt.trim().is_empty() {
            return Err(PyValueError::new_err("prompt cannot be empty"));
        }
        Ok(())
    }
}

#[pymethods]
impl LlmTools {
    fn llm_query(&self, prompt: String) -> PyResult<String> {
        Self::ensure_prompt(&prompt)?;
        self.reserve_calls(1)?;
        let response = self
            .runtime
            .block_on(async { self.lm.prompt(&prompt).await })
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(response)
    }

    fn llm_query_batched(&self, prompts: Vec<String>) -> PyResult<Vec<String>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        for prompt in &prompts {
            Self::ensure_prompt(prompt)?;
        }
        self.reserve_calls(prompts.len())?;

        let responses: Vec<_> = self.runtime.block_on(async {
            let futures: Vec<_> = prompts
                .iter()
                .map(|prompt| self.lm.prompt(prompt))
                .collect();
            futures::future::join_all(futures).await
        });

        let mut results = Vec::with_capacity(responses.len());
        for response in responses {
            match response {
                Ok(text) => results.push(text),
                Err(err) => return Err(PyRuntimeError::new_err(err.to_string())),
            }
        }
        Ok(results)
    }
}
