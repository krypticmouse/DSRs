use std::sync::{Arc, LazyLock, RwLock};

use super::LM;
use crate::adapter::Adapter;

pub struct Settings {
    pub lm: Arc<LM>,
}

impl Settings {
    /// The `adapter` parameter is accepted for API compatibility but unused —
    /// the typed path always formats with `ChatAdapter`, and no code path ever
    /// dispatched through the stored adapter.
    pub fn new(lm: LM, _adapter: impl Adapter + 'static) -> Self {
        Self { lm: Arc::new(lm) }
    }
}

pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn get_lm() -> Arc<LM> {
    Arc::clone(&GLOBAL_SETTINGS.read().unwrap().as_ref().unwrap().lm)
}

pub fn configure(lm: LM, adapter: impl Adapter + 'static) {
    let settings = Settings::new(lm, adapter);
    *GLOBAL_SETTINGS.write().unwrap() = Some(settings);
}
