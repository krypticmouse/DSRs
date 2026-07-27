use anyhow::Result;
use async_trait::async_trait;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile;
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};

use crate::{Prediction, RawExample};

/// Response-cache key: a 64-bit hash over the prompt + generation parameters.
///
/// Hashed keys keep foyer lookups and disk serialization O(1) in prompt size.
/// The hash is process-stable, which matches the cache's lifetime (the disk
/// tier lives in a per-process temp directory).
pub type CacheKey = u64;

/// A cached prompt-response pair.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The formatted prompt that was sent to the LM.
    pub prompt: String,
    /// The parsed prediction from the LM response.
    pub prediction: Prediction,
    /// The raw assistant text of the response, so [`LM::call`](crate::LM) can
    /// replay a cached completion through the normal parse path.
    #[serde(default)]
    pub raw_output: Option<String>,
}

/// Builds a deterministic cache key from a [`RawExample`].
///
/// `RawExample` is backed by a `HashMap`, whose iteration order is unstable —
/// pairs are sorted by field name before hashing so equal examples produce
/// equal keys.
fn normalized_key(key: RawExample) -> CacheKey {
    use std::hash::Hasher;

    let mut pairs: Vec<(String, Value)> = key.into_iter().collect();
    pairs.sort_by(|(left, _), (right, _)| left.cmp(right));

    struct FmtHasher<'a>(&'a mut std::hash::DefaultHasher);
    impl std::fmt::Write for FmtHasher<'_> {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.write(s.as_bytes());
            Ok(())
        }
    }

    let mut hasher = std::hash::DefaultHasher::new();
    for (name, value) in &pairs {
        hasher.write(name.as_bytes());
        use std::fmt::Write as _;
        let _ = write!(FmtHasher(&mut hasher), "{value:?}");
    }
    hasher.finish()
}

/// Interface for LM response caching.
///
/// Implemented by [`ResponseCache`]. The `insert` method takes a channel receiver
/// because the cache entry is produced asynchronously — the LM sends the entry
/// after the response is parsed, allowing the cache to be populated without
/// blocking the call return.
#[async_trait]
pub trait Cache: Send + Sync {
    async fn new() -> Self;
    async fn get(&self, key: RawExample) -> Result<Option<Prediction>>;
    async fn insert(&mut self, key: RawExample, rx: mpsc::Receiver<CacheEntry>) -> Result<()>;
    async fn get_history(&self, n: usize) -> Result<Vec<CacheEntry>>;
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

#[async_trait]
impl Cache for ResponseCache {
    #[tracing::instrument(name = "dsrs.cache.new", level = "debug")]
    async fn new() -> Self {
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

    #[tracing::instrument(
        name = "dsrs.cache.get",
        level = "trace",
        skip(self, key),
        fields(key_fields = key.data.len())
    )]
    async fn get(&self, key: RawExample) -> Result<Option<Prediction>> {
        Ok(self
            .get_entry(normalized_key(key))
            .await?
            .map(|entry| entry.prediction))
    }

    #[tracing::instrument(
        name = "dsrs.cache.insert",
        level = "trace",
        skip(self, key, rx),
        fields(key_fields = key.data.len(), window_size = self.window_size)
    )]
    async fn insert(&mut self, key: RawExample, mut rx: mpsc::Receiver<CacheEntry>) -> Result<()> {
        let Some(value) = rx.recv().await else {
            warn!("cache insert channel closed before receiving entry");
            return Ok(());
        };

        self.insert_entry(normalized_key(key), value);
        Ok(())
    }

    #[tracing::instrument(
        name = "dsrs.cache.get_history",
        level = "trace",
        skip(self),
        fields(n = n)
    )]
    async fn get_history(&self, n: usize) -> Result<Vec<CacheEntry>> {
        let actual_n = n.min(self.history_window.len());
        trace!(actual_n, "cache history fetched");
        Ok(self.history_window[..actual_n].to_vec())
    }
}

impl ResponseCache {
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
}
