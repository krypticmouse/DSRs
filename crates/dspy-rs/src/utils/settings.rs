use smart_default::SmartDefault;
use std::sync::{LazyLock, Mutex};

use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;

#[derive(SmartDefault, Clone)]
pub struct Settings {
    #[default(LM::default())]
    pub lm: LM,
    #[default(ChatAdapter {})]
    pub adapter: ChatAdapter,
}

impl Settings {
    pub fn configure(&mut self, lm: Option<LM>, adapter: Option<ChatAdapter>) {
        self.lm = lm.unwrap_or(self.lm.clone());
        self.adapter = adapter.unwrap_or(self.adapter.clone());
    }
}

pub static SETTINGS: LazyLock<Mutex<Settings>> = LazyLock::new(|| Mutex::new(Settings::default()));

pub fn configure_settings(lm: Option<LM>, adapter: Option<ChatAdapter>) {
    SETTINGS.lock().unwrap().configure(lm, adapter);
}
