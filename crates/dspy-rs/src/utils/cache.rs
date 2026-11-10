use std::sync::Arc;

use anyhow::Result;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tracing::{debug, trace, warn};

use crate::LmUsage;

/// Disk-tier initialization and lookup failures, with the backend cause intact.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("response cache {operation} failed: {source}")]
    Backend {
        operation: &'static str,
        #[source]
        source: anyhow::Error,
    },
}

impl CacheError {
    fn backend(operation: &'static str, source: impl Into<anyhow::Error>) -> Self {
        Self::Backend {
            operation,
            source: source.into(),
        }
    }
}

/// Response-cache key: a 64-bit hash over the prompt + generation parameters.
///
/// Hashed keys keep foyer lookups and disk serialization O(1) in prompt size.
/// The hash is process-stable, which matches the cache's lifetime (the disk
/// tier lives in a per-process temp directory). Keys are produced by
/// [`LM`](crate::LM) from the rendered chat — callers never build them by hand.
pub type CacheKey = u64;

const MEMORY_CAPACITY: usize = 256 * 1024 * 1024;
const DISK_CAPACITY: usize = 1024 * 1024 * 1024;

/// A cached prompt-response pair.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The formatted prompt that was sent to the LM.
    pub prompt: String,
    /// Token usage recorded for the original (uncached) call.
    #[serde(default)]
    pub usage: LmUsage,
    /// The raw assistant text of the response, so [`LM::call`](crate::LM) can
    /// replay a cached completion through the normal parse path.
    #[serde(default)]
    pub raw_output: Option<String>,
}

/// Hybrid memory + disk LM response cache.
///
/// Uses [foyer](https://docs.rs/foyer) with 256MB memory and 1GB disk (in a
/// temp directory owned by the cache for its whole lifetime). If the disk
/// tier cannot be initialized, the cache degrades to memory-only with a
/// warning instead of panicking. Execution history is owned independently
/// by [`LM`](crate::LM).
///
/// All methods take `&self`: the foyer cache is internally synchronized.
///
/// Created automatically by [`LM`](crate::LM) — you don't construct this directly.
#[derive(Clone)]
pub struct ResponseCache {
    handler: HybridCache<CacheKey, CacheEntry>,
    /// Keeps the disk tier's temp directory alive: `TempDir` deletes the
    /// directory on drop, so it must outlive every clone of the cache.
    /// `None` when running memory-only.
    _disk_dir: Option<Arc<TempDir>>,
}

impl ResponseCache {
    #[tracing::instrument(name = "dsrs.cache.new", level = "debug")]
    pub async fn new() -> Self {
        let (handler, disk_dir) = match Self::try_build_hybrid().await {
            Ok((handler, dir)) => (handler, Some(Arc::new(dir))),
            Err(error) => {
                warn!(
                    error = %error,
                    "disk cache tier unavailable; falling back to memory-only response cache"
                );
                (Self::build_memory_only().await, None)
            }
        };

        let cache = Self {
            handler,
            _disk_dir: disk_dir,
        };
        debug!(
            disk_tier = cache._disk_dir.is_some(),
            "response cache initialized"
        );
        cache
    }

    /// Builds the full memory + disk hybrid, returning the `TempDir` guard
    /// that must be held for as long as the cache lives.
    async fn try_build_hybrid() -> Result<(HybridCache<CacheKey, CacheEntry>, TempDir)> {
        let dir =
            tempfile::tempdir().map_err(|error| CacheError::backend("create directory", error))?;

        let device = FsDeviceBuilder::new(dir.path())
            .with_capacity(DISK_CAPACITY)
            .build()
            .map_err(|error| CacheError::backend("create device", error))?;

        let hybrid = HybridCacheBuilder::new()
            .memory(MEMORY_CAPACITY)
            .storage()
            .with_engine_config(BlockEngineBuilder::new(device))
            .build()
            .await
            .map_err(|error| CacheError::backend("initialize disk tier", error))?;
        Ok((hybrid, dir))
    }

    /// Memory-only fallback: foyer's storage phase defaults to a noop engine,
    /// which cannot fail to build (no I/O involved).
    async fn build_memory_only() -> HybridCache<CacheKey, CacheEntry> {
        HybridCacheBuilder::new()
            .memory(MEMORY_CAPACITY)
            .storage()
            .build()
            .await
            .expect("memory-only foyer cache construction cannot fail")
    }

    /// Fetches the full cached entry (including raw output) for a key.
    #[tracing::instrument(name = "dsrs.cache.get_entry", level = "trace", skip(self))]
    pub async fn get_entry(&self, key: CacheKey) -> Result<Option<CacheEntry>> {
        let value = self
            .handler
            .get(&key)
            .await
            .map_err(|error| CacheError::backend("read", error))?
            .map(|v| v.value().clone());
        trace!(hit = value.is_some(), "cache lookup complete");
        Ok(value)
    }

    /// Replays only the response, without copying the stored prompt and usage.
    #[tracing::instrument(name = "dsrs.cache.get_response", level = "trace", skip(self))]
    pub(crate) async fn get_response(&self, key: CacheKey) -> Result<Option<String>> {
        let response = self
            .handler
            .get(&key)
            .await
            .map_err(|error| CacheError::backend("read", error))?
            .and_then(|entry| entry.value().raw_output.clone());
        trace!(hit = response.is_some(), "cache lookup complete");
        Ok(response)
    }

    /// Inserts an entry synchronously — the direct path used by [`LM::call`](crate::LM).
    #[tracing::instrument(name = "dsrs.cache.insert_entry", level = "trace", skip(self, entry))]
    pub fn insert_entry(&self, key: CacheKey, entry: CacheEntry) {
        let prompt_len = entry.prompt.len();
        self.handler.insert(key, entry);
        trace!(prompt_len, "cache entry inserted");
    }
}
