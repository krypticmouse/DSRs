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
