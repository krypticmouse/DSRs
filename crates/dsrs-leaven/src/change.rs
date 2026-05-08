#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct DsrsProgramChange {
    pub address: String,
    pub replacement: serde_json::Value,
}
