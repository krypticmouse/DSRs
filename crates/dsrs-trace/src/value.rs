use serde_json::Value;

pub use dsrs_core::TrackedValue;

pub trait IntoTracked {
    fn into_tracked(self) -> TrackedValue;
}

impl IntoTracked for TrackedValue {
    fn into_tracked(self) -> Self {
        self
    }
}

impl IntoTracked for Value {
    fn into_tracked(self) -> TrackedValue {
        TrackedValue {
            value: self,
            source: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_value_becomes_unlinked_tracked_value() {
        let tracked = serde_json::json!({"answer": 42}).into_tracked();
        assert_eq!(tracked.value["answer"], 42);
        assert!(tracked.source.is_none());
    }

    #[test]
    fn tracked_value_identity_conversion_preserves_source() {
        let original = TrackedValue {
            value: serde_json::json!("x"),
            source: Some((3, "field".to_string())),
        };
        let tracked = original.clone().into_tracked();
        assert_eq!(tracked.value, original.value);
        assert_eq!(tracked.source, original.source);
    }
}
