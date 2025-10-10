use anyhow::Result;
use async_trait::async_trait;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde_json::Value;
use tempfile;

use crate::{Example, Prediction};

#[async_trait]
pub trait Cache: Send + Sync {
    async fn new() -> Self;
    async fn get(&self, key: Example) -> Result<Option<Prediction>>;
    fn insert(&self, key: Example, value: Prediction) -> Result<()>;
}

#[derive(Clone)]
pub struct ResponseCache {
    handler: HybridCache<Vec<(String, Value)>, Vec<(String, Value)>>,
}

#[async_trait]
impl Cache for ResponseCache {
    async fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();

        let device = FsDeviceBuilder::new(dir.path())
            .with_capacity(256 * 1024 * 1024)
            .build()
            .unwrap();

        let hybrid: HybridCache<Vec<(String, Value)>, Vec<(String, Value)>> =
            HybridCacheBuilder::new()
                .memory(64 * 1024 * 1024)
                .storage()
                .with_engine_config(BlockEngineBuilder::new(device))
                .build()
                .await
                .unwrap();
        Self { handler: hybrid }
    }

    async fn get(&self, key: Example) -> Result<Option<Prediction>> {
        let key = key.into_iter().collect::<Vec<(String, Value)>>();

        let value = self.handler.get(&key).await?.map(|v| v.value().clone());

        Ok(value.map(Prediction::from))
    }

    fn insert(&self, key: Example, value: Prediction) -> Result<()> {
        let key = key.into_iter().collect::<Vec<(String, Value)>>();
        let value = value.into_iter().collect::<Vec<(String, Value)>>();
        self.handler.insert(key, value);

        Ok(())
    }
}
