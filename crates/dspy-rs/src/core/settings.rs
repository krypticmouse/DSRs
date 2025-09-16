use std::sync::{Arc, LazyLock, RwLock};

use super::LM;
use crate::adapter::Adapter;

pub struct Settings<'a> {
    pub lm: &'a LM,
    pub adapter: Arc<dyn Adapter>,
}

pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings<'static>>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn get_lm() -> &'static LM {
    GLOBAL_SETTINGS
        .read()
        .unwrap()
        .as_ref()
        .unwrap()
        .lm
}

pub fn configure(lm: &LM, adapter: impl Adapter + 'static) {
    let static_lm: &'static LM = Box::leak(Box::new(lm));
    let settings = Settings {
        lm: static_lm,
        adapter: Arc::new(adapter),
    };
    *GLOBAL_SETTINGS.write().unwrap() = Some(settings);
}
