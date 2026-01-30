//! Renderer definitions for prompt formatting.

use std::{error::Error, fmt};

use baml_types::StreamingMode;
use indexmap::IndexMap;
use minijinja::Value;
use serde::Serialize;

use super::PromptValue;

#[derive(Debug, Clone)]
pub struct PromptRenderer;

#[derive(Debug, Clone)]
pub struct RenderSettings {
    /// Total output budget across the full render.
    pub max_total_chars: usize,
    /// Per-string truncation limit.
    pub max_string_chars: usize,
    /// Iteration cap for list rendering.
    pub max_list_items: usize,
    /// Iteration cap for map/class rendering.
    pub max_map_entries: usize,
    /// Recursion depth limit.
    pub max_depth: usize,
    /// Max number of union branches shown in summaries like "A | B | C".
    pub max_union_branches_shown: usize,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            max_total_chars: 50_000,
            max_string_chars: 5_000,
            max_list_items: 100,
            max_map_entries: 50,
            max_depth: 10,
            max_union_branches_shown: 5,
        }
    }
}

/// Per-render context (one per render_messages call).
#[derive(Debug, Clone)]
pub struct RenderSession {
    /// Settings (can override world defaults).
    pub settings: RenderSettings,
    /// Custom template context (passed to templates as `ctx`).
    pub ctx: Value,
    /// Current recursion depth.
    pub depth: usize,
    /// Recursion guard stack: (TypeKey, path) pairs.
    pub stack: Vec<(TypeKey, String)>,
}

impl RenderSession {
    pub fn new(settings: RenderSettings) -> Self {
        Self {
            settings,
            ctx: Value::UNDEFINED,
            depth: 0,
            stack: Vec::new(),
        }
    }

    pub fn with_ctx<T: Serialize>(mut self, ctx: T) -> Self {
        self.ctx = Value::from_serialize(&ctx);
        self
    }

    pub fn push_depth(&self) -> Self {
        let mut new = self.clone();
        new.depth += 1;
        new
    }

    pub fn check_recursion(&self, key: &TypeKey) -> bool {
        self.stack.iter().any(|(k, _)| k == key)
    }
}

/// Key for looking up renderers in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RendererKey {
    pub type_key: TypeKey,
    pub style: &'static str,
}

/// Identifies a type for renderer lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKey {
    Class { name: String, mode: StreamingMode },
    Enum { name: String },
}

impl RendererKey {
    pub fn for_class(name: impl Into<String>, mode: StreamingMode, style: &'static str) -> Self {
        Self {
            type_key: TypeKey::Class {
                name: name.into(),
                mode,
            },
            style,
        }
    }

    pub fn for_enum(name: impl Into<String>, style: &'static str) -> Self {
        Self {
            type_key: TypeKey::Enum { name: name.into() },
            style,
        }
    }
}

#[derive(Debug)]
pub struct RenderError {
    pub path: String,
    pub ty: String,
    pub style: String,
    pub renderer: String,
    pub template_name: Option<String>,
    pub template_location: Option<(usize, usize)>,
    pub message: String,
    pub cause: Option<Box<dyn Error + Send + Sync>>,
}

/// Result type for render operations.
pub type RenderResult = Result<String, RenderError>;

/// Specification for a renderer (pre-compilation).
pub enum RendererSpec {
    /// Jinja template source.
    Jinja { source: &'static str },
    /// Function-based renderer.
    Func {
        f: fn(&PromptValue, &RenderSession) -> RenderResult,
    },
}

/// Container for collected renderer specs from registry.
#[derive(Debug, Default)]
pub struct RendererDbSeed {
    pub specs: IndexMap<RendererKey, RendererSpec>,
}

impl RendererDbSeed {
    pub fn new() -> Self {
        Self {
            specs: IndexMap::new(),
        }
    }

    pub fn insert(&mut self, key: RendererKey, spec: RendererSpec) {
        self.specs.insert(key, spec);
    }
}

impl RenderError {
    pub fn new(
        path: impl Into<String>,
        ty: impl Into<String>,
        style: impl Into<String>,
        renderer: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            ty: ty.into(),
            style: style.into(),
            renderer: renderer.into(),
            template_name: None,
            template_location: None,
            message: message.into(),
            cause: None,
        }
    }

    pub fn template_error(
        path: impl Into<String>,
        ty: impl Into<String>,
        style: impl Into<String>,
        renderer: impl Into<String>,
        template_name: impl Into<String>,
        template_location: Option<(usize, usize)>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(path, ty, style, renderer, message)
            .with_template(template_name, template_location)
    }

    pub fn with_template(
        mut self,
        template_name: impl Into<String>,
        template_location: Option<(usize, usize)>,
    ) -> Self {
        self.template_name = Some(template_name.into());
        self.template_location = template_location;
        self
    }

    pub fn with_cause<E>(mut self, cause: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        self.cause = Some(Box::new(cause));
        self
    }

    fn template_summary(&self) -> Option<String> {
        match (&self.template_name, self.template_location) {
            (None, None) => None,
            (Some(name), None) => Some(name.clone()),
            (Some(name), Some((line, column))) => Some(format!("{name}:{line}:{column}")),
            (None, Some((line, column))) => Some(format!("<unknown>:{line}:{column}")),
        }
    }
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Render failed at '{}'", self.path)?;
        writeln!(f, "  Type: {}", self.ty)?;
        writeln!(f, "  Style: {}", self.style)?;
        writeln!(f, "  Renderer: {}", self.renderer)?;
        if let Some(template) = self.template_summary() {
            writeln!(f, "  Template: {}", template)?;
        }
        if let Some(cause) = self.cause.as_deref() {
            writeln!(f, "  Error: {}", self.message)?;
            write!(f, "  Cause: {cause}")?;
        } else {
            write!(f, "  Error: {}", self.message)?;
        }
        Ok(())
    }
}

impl Error for RenderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.cause
            .as_deref()
            .map(|cause| cause as &(dyn Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::RenderError;
    use std::io;

    #[test]
    fn formats_basic_render_error() {
        let err = RenderError::new(
            "inputs.history.entries[3].output",
            "REPLEntry",
            "json",
            "type:REPLEntry:json",
            "undefined field 'outputs'",
        );

        let expected = concat!(
            "Render failed at 'inputs.history.entries[3].output'\n",
            "  Type: REPLEntry\n",
            "  Style: json\n",
            "  Renderer: type:REPLEntry:json\n",
            "  Error: undefined field 'outputs'",
        );

        assert_eq!(err.to_string(), expected);
    }

    #[test]
    fn formats_template_render_error_with_cause() {
        let err = RenderError::template_error(
            "inputs.history.entries[3].output",
            "REPLEntry",
            "json",
            "type:REPLEntry:json",
            "repl_entry.jinja",
            Some((15, 8)),
            "undefined field 'outputs'",
        )
        .with_cause(io::Error::new(io::ErrorKind::Other, "boom"));

        let expected = concat!(
            "Render failed at 'inputs.history.entries[3].output'\n",
            "  Type: REPLEntry\n",
            "  Style: json\n",
            "  Renderer: type:REPLEntry:json\n",
            "  Template: repl_entry.jinja:15:8\n",
            "  Error: undefined field 'outputs'\n",
            "  Cause: boom",
        );

        assert_eq!(err.to_string(), expected);
    }
}
