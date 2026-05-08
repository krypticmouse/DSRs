#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct DsrsEvidence {
    pub payload: serde_json::Value,
}

impl leaven_core::Evidence for DsrsEvidence {}
