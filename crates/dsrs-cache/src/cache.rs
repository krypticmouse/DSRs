use anyhow::Result;
use async_trait::async_trait;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile;
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};

use dsrs_core::{Prediction, RawExample};

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

#[cfg(test)]
mod tests {
    use super::*;
    use dsrs_core::{LmUsage, hashmap};

    fn raw_key(question: &str) -> RawExample {
        RawExample::new(
            hashmap! {
                "question".to_string() => question.into(),
            },
            vec!["question".to_string()],
            vec![],
        )
    }

    fn prediction(answer: &str) -> Prediction {
        Prediction::new(
            hashmap! {
                "answer".to_string() => answer.into(),
            },
            LmUsage::default(),
        )
    }

    #[tokio::test]
    async fn insert_get_and_history_round_trip_cached_prediction() {
        let mut cache = ResponseCache::new().await;
        let key = raw_key("capital?");
        assert!(cache.get(key.clone()).await.unwrap().is_none());

        let (tx, rx) = mpsc::channel(1);
        let entry = CacheEntry {
            prompt: "prompt".to_string(),
            prediction: prediction("Paris"),
        };
        tx.send(entry.clone()).await.unwrap();
        drop(tx);

        cache.insert(key.clone(), rx).await.unwrap();

        let cached = cache.get(key).await.unwrap().unwrap();
        assert_eq!(cached.get("answer", None), "Paris");
        let history = cache.get_history(10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].prompt, entry.prompt);
    }

    #[tokio::test]
    async fn insert_with_closed_channel_is_noop() {
        let mut cache = ResponseCache::new().await;
        let (_tx, rx) = mpsc::channel(1);
        drop(_tx);

        cache.insert(raw_key("missing"), rx).await.unwrap();

        assert!(cache.get_history(1).await.unwrap().is_empty());
    }
}
