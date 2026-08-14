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

pub fn get_lm() -> Arc<LM> {
    Arc::clone(&GLOBAL_SETTINGS.read().unwrap().as_ref().unwrap().lm)
}

pub fn configure(lm: LM) {
    *GLOBAL_SETTINGS.write().unwrap() = Some(Settings::new(lm));
}
