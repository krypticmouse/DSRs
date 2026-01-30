//! Renderer definitions for prompt formatting.

use std::{error::Error, fmt};

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
