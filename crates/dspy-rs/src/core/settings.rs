use std::sync::{Arc, LazyLock, RwLock};

use super::LM;

pub struct Settings {
    pub lm: Arc<LM>,
}

impl Settings {
    pub fn new(lm: LM) -> Self {
        Self { lm: Arc::new(lm) }
    }
}

pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings>>> =
    LazyLock::new(|| RwLock::new(None));

tokio::task_local! {
    static SCOPED_LM: Arc<LM>;
    static CALL_LM: Arc<LM>;
}

/// Runs a future with a task-local default LM. Predictor-specific LMs take
/// precedence. Nested scopes restore automatically on return, error, panic, or
/// cancellation. Spawned tasks do not inherit the scope.
pub async fn with_lm<F: std::future::Future>(lm: impl Into<Arc<LM>>, future: F) -> F::Output {
    SCOPED_LM.scope(lm.into(), future).await
}

pub(crate) async fn with_call_lm<F: std::future::Future>(
    lm: impl Into<Arc<LM>>,
    future: F,
) -> F::Output {
    CALL_LM.scope(lm.into(), future).await
}

pub(crate) fn call_lm() -> Option<Arc<LM>> {
    CALL_LM.try_with(Arc::clone).ok()
}

/// Returns the active call LM, scoped default, or global LM, if configured.
pub fn try_get_lm() -> Option<Arc<LM>> {
    call_lm()
        .or_else(|| SCOPED_LM.try_with(Arc::clone).ok())
        .or_else(|| {
            GLOBAL_SETTINGS
                .read()
                .unwrap()
                .as_ref()
                .map(|settings| Arc::clone(&settings.lm))
        })
}

/// Returns the active call LM, scoped default, or global LM.
///
/// Panics if none is configured.
pub fn get_lm() -> Arc<LM> {
    try_get_lm().expect("No LM configured: use configure(), with_lm(), or a predictor LM")
}

pub fn configure(lm: LM) {
    *GLOBAL_SETTINGS.write().unwrap() = Some(Settings::new(lm));
}
