use anyhow::Result;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde::{Deserialize, Serialize};
use tempfile;
use tracing::{debug, trace};

use crate::LmUsage;

/// Response-cache key: a 64-bit hash over the prompt + generation parameters.
///
/// Hashed keys keep foyer lookups and disk serialization O(1) in prompt size.
/// The hash is process-stable, which matches the cache's lifetime (the disk
/// tier lives in a per-process temp directory). Keys are produced by
/// [`LM`](crate::LM) from the rendered chat — callers never build them by hand.
pub type CacheKey = u64;

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
/// temp directory). Maintains a sliding window of the 100 most recent entries
/// for [`inspect_history`](crate::LM::inspect_history).
///
/// Created automatically by [`LM`](crate::LM) — you don't construct this directly.
#[derive(Clone)]
pub struct ResponseCache {
    handler: HybridCache<CacheKey, CacheEntry>,
    window_size: usize,
    history_window: Vec<CacheEntry>,
}

impl ResponseCache {
    #[tracing::instrument(name = "dsrs.cache.new", level = "debug")]
    pub async fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();

        let device = FsDeviceBuilder::new(dir.path())
            .with_capacity(1024 * 1024 * 1024)
            .build()
            .unwrap();

        let hybrid: HybridCache<CacheKey, CacheEntry> = HybridCacheBuilder::new()
            .memory(256 * 1024 * 1024)
            .storage()
            .with_engine_config(BlockEngineBuilder::new(device))
            .build()
            .await
            .unwrap();
        let cache = Self {
            handler: hybrid,
            window_size: 100,
            history_window: Vec::new(),
        };
        debug!(
            window_size = cache.window_size,
            "response cache initialized"
        );
        cache
    }

    /// Fetches the full cached entry (including raw output) for a key.
    #[tracing::instrument(name = "dsrs.cache.get_entry", level = "trace", skip(self))]
    pub async fn get_entry(&self, key: CacheKey) -> Result<Option<CacheEntry>> {
        let value = self.handler.get(&key).await?.map(|v| v.value().clone());
        trace!(hit = value.is_some(), "cache lookup complete");
        Ok(value)
    }

    /// Inserts an entry synchronously — the direct path used by [`LM::call`](crate::LM).
    #[tracing::instrument(
        name = "dsrs.cache.insert_entry",
        level = "trace",
        skip(self, entry),
        fields(window_size = self.window_size)
    )]
    pub fn insert_entry(&mut self, key: CacheKey, entry: CacheEntry) {
        self.history_window.insert(0, entry.clone());
        if self.history_window.len() > self.window_size {
            self.history_window.pop();
        }
        self.handler.insert(key, entry.clone());
        trace!(
            history_len = self.history_window.len(),
            prompt_len = entry.prompt.len(),
            "cache entry inserted"
        );
    }

    /// Returns the `n` most recent cached entries (newest first).
    #[tracing::instrument(
        name = "dsrs.cache.get_history",
        level = "trace",
        skip(self),
        fields(n = n)
    )]
    pub async fn get_history(&self, n: usize) -> Result<Vec<CacheEntry>> {
        let actual_n = n.min(self.history_window.len());
        trace!(actual_n, "cache history fetched");
        Ok(self.history_window[..actual_n].to_vec())
    }
}
