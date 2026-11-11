//! Persistent invocation storage, independent of caching and history retention.
use crate::HistoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("interaction storage I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid interaction journal: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported interaction journal version {0}")]
    Version(u32),
    #[error("interaction persistence failed: {0}")]
    Session(String),
}

/// Parsing upserts the same invocation ID rather than creating another call.
pub trait InteractionStore: std::fmt::Debug + Send + Sync {
    fn upsert(&self, entry: &HistoryEntry) -> Result<(), StorageError>;
    fn get(&self, id: &str) -> Result<Option<HistoryEntry>, StorageError>;
    /// Newest invocations first, with at most one revision per ID.
    fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>, StorageError>;
    fn upsert_trace(&self, trace: &crate::Trace) -> Result<(), StorageError>;
    fn get_trace(&self, id: &str) -> Result<Option<crate::Trace>, StorageError>;
}

/// Versioned, append-only JSONL with durable updates and OS file locking.
/// Corrupt complete records fail reads; a torn final line is repaired on append.
/// Contains prompts/responses: choose a path with appropriate access permissions.
#[derive(Debug)]
pub struct JsonlInteractionStore {
    file: Mutex<File>,
}

#[derive(Serialize, Deserialize)]
struct Revision {
    version: u32,
    #[serde(flatten)]
    record: Record,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Record {
    Invocation { entry: HistoryEntry },
    Execution { trace: crate::Trace },
}

struct LockedFile<'a>(&'a mut File);
impl Drop for LockedFile<'_> {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

impl JsonlInteractionStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let store = Self {
            file: Mutex::new(file),
        };
        store.recent(0)?;
        Ok(store)
    }

    fn locked<T>(
        &self,
        f: impl FnOnce(&mut File) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut file = self.file.lock().unwrap_or_else(|p| p.into_inner());
        file.lock()?;
        let guard = LockedFile(&mut file);
        f(guard.0)
    }

    fn read(
        file: &mut File,
    ) -> Result<
        (
            HashMap<String, HistoryEntry>,
            HashMap<String, crate::Trace>,
            u64,
        ),
        StorageError,
    > {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let complete = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let mut entries = HashMap::new();
        let mut traces = HashMap::new();
        for line in bytes[..complete]
            .split(|b| *b == b'\n')
            .filter(|line| !line.is_empty())
        {
            let revision: Revision = serde_json::from_slice(line)?;
            if revision.version != 1 {
                return Err(StorageError::Version(revision.version));
            }
            match revision.record {
                Record::Invocation { entry } => {
                    entries.insert(entry.id.clone(), entry);
                }
                Record::Execution { trace } => {
                    traces.insert(trace.meta.trace_id.clone(), trace);
                }
            }
        }
        Ok((entries, traces, complete as u64))
    }

    fn append(&self, record: Record) -> Result<(), StorageError> {
        let mut bytes = serde_json::to_vec(&Revision { version: 1, record })?;
        bytes.push(b'\n');
        self.locked(|file| {
            let (_, _, complete) = Self::read(file)?;
            file.set_len(complete)?;
            file.seek(SeekFrom::Start(complete))?;
            file.write_all(&bytes)?;
            file.sync_data()?;
            Ok(())
        })
    }
}

impl InteractionStore for JsonlInteractionStore {
    fn upsert(&self, entry: &HistoryEntry) -> Result<(), StorageError> {
        self.append(Record::Invocation {
            entry: entry.clone(),
        })
    }
    fn upsert_trace(&self, trace: &crate::Trace) -> Result<(), StorageError> {
        self.append(Record::Execution {
            trace: trace.clone(),
        })
    }
    fn get_trace(&self, id: &str) -> Result<Option<crate::Trace>, StorageError> {
        self.locked(|file| Ok(Self::read(file)?.1.remove(id)))
    }
    fn get(&self, id: &str) -> Result<Option<HistoryEntry>, StorageError> {
        self.locked(|file| Ok(Self::read(file)?.0.remove(id)))
    }
    fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>, StorageError> {
        self.locked(|file| {
            let mut entries: Vec<_> = Self::read(file)?.0.into_values().collect();
            entries.sort_by(|a, b| {
                b.timestamp_us
                    .cmp(&a.timestamp_us)
                    .then_with(|| b.id.cmp(&a.id))
            });
            entries.truncate(limit);
            Ok(entries)
        })
    }
}

#[derive(Debug)]
pub(crate) struct StoreSession {
    store: Arc<dyn InteractionStore>,
    errors: Mutex<Vec<String>>,
}
impl StoreSession {
    fn persist_trace(&self, trace: &crate::Trace) {
        if let Err(error) = self.store.upsert_trace(trace) {
            tracing::error!(%error, trace_id = %trace.meta.trace_id, "trace persistence failed");
            self.errors
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(error.to_string());
        }
    }
    pub(crate) fn persist(&self, entry: &HistoryEntry) {
        if let Err(error) = self.store.upsert(entry) {
            tracing::error!(%error, call_id = %entry.id, "interaction persistence failed");
            self.errors
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(error.to_string());
        }
    }
}
tokio::task_local! { static STORES: Vec<Arc<StoreSession>>; }
pub(crate) fn current_stores() -> Vec<Arc<StoreSession>> {
    STORES.try_with(Clone::clone).unwrap_or_default()
}

pub(crate) fn persist_trace(trace: &crate::Trace) {
    for session in current_stores() {
        session.persist_trace(trace);
    }
}

/// Persist calls, cache hits, failures, cancellation and subsequent parse results.
/// History capacity may be zero. Nested scopes compose; spawned tasks need their
/// own scope. Storage errors are returned after the future finishes (provider
/// errors remain inside T). Writes are synchronous and durable before return.
pub async fn with_interaction_store<T>(
    store: Arc<dyn InteractionStore>,
    future: impl Future<Output = T>,
) -> Result<T, StorageError> {
    let session = Arc::new(StoreSession {
        store,
        errors: Mutex::new(Vec::new()),
    });
    let mut stores = current_stores();
    stores.push(session.clone());
    let output = STORES.scope(stores, future).await;
    let errors = session.errors.lock().unwrap_or_else(|p| p.into_inner());
    if errors.is_empty() {
        Ok(output)
    } else {
        Err(StorageError::Session(errors.join("; ")))
    }
}
