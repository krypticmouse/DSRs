use std::fmt;

/// Stable dotted path to a DSRs predictor leaf discovered in a module.
#[derive(
    Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Deserialize, serde::Serialize,
)]
#[serde(transparent)]
pub struct PredictorPath(String);

impl PredictorPath {
    /// Creates a predictor path from the DSRs Facet discovery path.
    ///
    /// Empty paths are rejected because the bridge uses paths as durable surface
    /// part ids, addresses, and snapshot-map keys.
    pub fn new(path: impl Into<String>) -> Result<Self, crate::DsrsLeavenError> {
        let path = path.into();
        if path.trim().is_empty() {
            return Err(crate::DsrsLeavenError::InvalidPredictorPath(path));
        }
        Ok(Self(path))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PredictorPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for PredictorPath {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Edit to one DSRs predictor snapshot.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum DsrsPredictorEdit {
    /// Replace the effective predictor instruction.
    ReplaceInstruction(String),
}

/// Artifact-native change for a DSRs program snapshot.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct DsrsProgramChange {
    path: PredictorPath,
    edit: DsrsPredictorEdit,
}

impl DsrsProgramChange {
    #[must_use]
    pub fn new(path: PredictorPath, edit: DsrsPredictorEdit) -> Self {
        Self { path, edit }
    }

    #[must_use]
    pub fn replace_instruction(path: PredictorPath, instruction: String) -> Self {
        Self::new(path, DsrsPredictorEdit::ReplaceInstruction(instruction))
    }

    #[must_use]
    pub fn path(&self) -> &PredictorPath {
        &self.path
    }

    #[must_use]
    pub fn edit(&self) -> &DsrsPredictorEdit {
        &self.edit
    }
}
