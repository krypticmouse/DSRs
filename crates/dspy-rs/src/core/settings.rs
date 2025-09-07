use std::sync::{Arc, LazyLock, RwLock};

use super::LM;
use crate::adapter::Adapter;

pub struct Settings {
    pub lm: LM,
    pub adapter: Arc<dyn Adapter>,
}

pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn configure(lm: LM, adapter: impl Adapter + 'static) {
    let settings = Settings {
        lm,
        adapter: Arc::new(adapter),
    };
    *GLOBAL_SETTINGS.write().unwrap() = Some(settings);
}
