use std::marker::PhantomData;
use std::sync::Arc;

use serde::Serialize;

use crate::baml_bridge::prompt::{PromptWorld, RenderError};

use super::{SigMeta, Signature};

/// A compiled signature ready for prompt rendering.
pub struct CompiledSignature<S: Signature> {
    /// The prompt world with types and renderers.
    pub world: Arc<PromptWorld>,
    /// System message template (compiled name in env).
    pub system_template: String,
    /// User message template (compiled name in env).
    pub user_template: String,
    /// Signature metadata for templates.
    pub sig_meta: SigMeta,
    pub(crate) _phantom: PhantomData<S>,
}

/// Rendered prompt messages.
pub struct RenderedMessages {
    pub system: String,
    pub user: String,
}

impl<S: Signature> CompiledSignature<S> {
    /// Render messages with default settings.
    pub fn render_messages(
        &self,
        input: &S::Input,
    ) -> Result<RenderedMessages, Box<RenderError>> {
        self.render_messages_with_ctx(input, ())
    }

    /// Render messages with custom context.
    pub fn render_messages_with_ctx<C: Serialize>(
        &self,
        _input: &S::Input,
        _ctx: C,
    ) -> Result<RenderedMessages, Box<RenderError>> {
        todo!("render_messages_with_ctx implemented in dsrs-n9u.40")
    }
}
