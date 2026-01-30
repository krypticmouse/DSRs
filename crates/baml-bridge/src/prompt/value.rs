//! Prompt value wrappers for typed rendering.

use std::fmt;

#[derive(Debug, Clone, Default)]
pub struct PromptPath {
    segments: Vec<PathSegment>,
}

#[derive(Debug, Clone)]
enum PathSegment {
    Field(String),
    Index(usize),
    MapKey(String),
}

impl PromptPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_field(&self, name: impl Into<String>) -> Self {
        let mut new = self.clone();
        new.segments.push(PathSegment::Field(name.into()));
        new
    }

    pub fn push_index(&self, idx: usize) -> Self {
        let mut new = self.clone();
        new.segments.push(PathSegment::Index(idx));
        new
    }

    pub fn push_map_key(&self, key: impl Into<String>) -> Self {
        let mut new = self.clone();
        new.segments.push(PathSegment::MapKey(key.into()));
        new
    }
}

impl fmt::Display for PromptPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for segment in &self.segments {
            match segment {
                PathSegment::Field(name) => {
                    if first {
                        write!(f, "{name}")?;
                    } else {
                        write!(f, ".{name}")?;
                    }
                }
                PathSegment::Index(idx) => {
                    write!(f, "[{idx}]")?;
                }
                PathSegment::MapKey(key) => {
                    let escaped = key.replace('\\', "\\\\").replace('"', "\\\"");
                    write!(f, "[\"{escaped}\"]")?;
                }
            }
            first = false;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PromptValue;

#[cfg(test)]
mod tests {
    use super::PromptPath;

    #[test]
    fn formats_field_and_index_path() {
        let path = PromptPath::new()
            .push_field("inputs")
            .push_field("history")
            .push_field("entries")
            .push_index(3)
            .push_field("output");

        assert_eq!(path.to_string(), "inputs.history.entries[3].output");
    }

    #[test]
    fn formats_map_key_path() {
        let path = PromptPath::new().push_field("meta").push_map_key("key");

        assert_eq!(path.to_string(), "meta[\"key\"]");
    }
}
