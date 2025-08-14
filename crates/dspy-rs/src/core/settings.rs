use std::sync::{LazyLock, RwLock};

use super::LM;
use crate::adapter::ConcreteAdapter;

pub struct Settings {
    pub lm: LM,
    pub adapter: ConcreteAdapter,
}

pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn configure(lm: LM, adapter: ConcreteAdapter) {
    let settings = Settings { lm, adapter };
    *GLOBAL_SETTINGS.write().unwrap() = Some(settings);
}
