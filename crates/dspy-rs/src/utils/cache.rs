use anyhow::Result;
use async_trait::async_trait;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile;
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};

use crate::{Prediction, RawExample};

type CacheKey = Vec<(String, Value)>;

/// A cached prompt-response pair.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The formatted prompt that was sent to the LM.
    pub prompt: String,
    /// The parsed prediction from the LM response.
    pub prediction: Prediction,
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
        let key = key.into_iter().collect::<CacheKey>();

        let value = self.handler.get(&key).await?.map(|v| v.value().clone());
        trace!(hit = value.is_some(), "cache lookup complete");

        Ok(value.map(|entry| entry.prediction))
    }

    #[tracing::instrument(
        name = "dsrs.cache.insert",
        level = "trace",
        skip(self, key, rx),
        fields(key_fields = key.data.len(), window_size = self.window_size)
    )]
    async fn insert(&mut self, key: RawExample, mut rx: mpsc::Receiver<CacheEntry>) -> Result<()> {
        let key = key.into_iter().collect::<CacheKey>();
        let Some(value) = rx.recv().await else {
            warn!("cache insert channel closed before receiving entry");
            return Ok(());
        };

        self.history_window.insert(0, value.clone());
        if self.history_window.len() > self.window_size {
            self.history_window.pop();
        }
        self.handler.insert(key, value.clone());
        trace!(
            history_len = self.history_window.len(),
            prompt_len = value.prompt.len(),
            "cache entry inserted"
        );

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
