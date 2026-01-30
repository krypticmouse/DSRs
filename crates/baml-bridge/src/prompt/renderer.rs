//! Renderer definitions for prompt formatting.

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
