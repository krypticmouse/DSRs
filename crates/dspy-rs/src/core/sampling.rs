//! Scoped sampling identity and per-call generation overrides.
//!
//! Sampling state is task-local so concurrent rollouts cannot affect one
//! another. Nested work can derive stable child identities for retries or
//! refinement attempts.

use tokio::task_local;

task_local! {
    static CURRENT: SamplingOptions;
}

/// Sampling settings for one logical rollout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingOptions {
    /// Stable identity of this sample. Equal identities intentionally share
    /// LM response cache entries.
    pub id: Option<String>,
    /// Optional per-scope temperature override.
    pub temperature: Option<f32>,
}

impl SamplingOptions {
    /// Returns an options value for a deterministic child operation.
    pub fn child(&self, component: impl AsRef<str>, attempt: usize) -> Self {
        let parent = self.id.as_deref().unwrap_or("sample");
        let mut child = self.clone();
        child.id = Some(format!("{parent}/{}/{}", component.as_ref(), attempt));
        child
    }
}

/// Reads the active task-local settings, or defaults outside a scope.
pub fn current_sampling_options() -> SamplingOptions {
    CURRENT.try_with(Clone::clone).unwrap_or_default()
}

/// Runs `future` under task-local sampling settings. The enclosing settings
/// are restored automatically when the future completes or is cancelled.
pub async fn with_sampling_options<F: std::future::Future>(
    options: SamplingOptions,
    future: F,
) -> F::Output {
    CURRENT.scope(options, future).await
}

/// Returns the effective temperature for an LM configuration.
pub(crate) fn effective_temperature(configured: f32) -> f32 {
    current_sampling_options().temperature.unwrap_or(configured)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn nested_scope_restores_parent_settings() {
        let parent = SamplingOptions {
            id: Some("rollout".into()),
            temperature: Some(0.4),
        };
        with_sampling_options(parent.clone(), async {
            assert_eq!(current_sampling_options(), parent);
            let child = current_sampling_options().child("refine", 1);
            with_sampling_options(child.clone(), async {
                assert_eq!(current_sampling_options(), child);
            })
            .await;
            assert_eq!(current_sampling_options(), parent);
        })
        .await;
        assert_eq!(current_sampling_options(), SamplingOptions::default());
    }

    #[tokio::test]
    async fn parallel_scopes_do_not_contaminate_each_other() {
        async fn read(id: &'static str, temperature: f32) -> (Option<String>, f32) {
            with_sampling_options(
                SamplingOptions {
                    id: Some(id.into()),
                    temperature: Some(temperature),
                },
                async {
                    tokio::task::yield_now().await;
                    let options = current_sampling_options();
                    (options.id, effective_temperature(0.7))
                },
            )
            .await
        }

        let (first, second) = tokio::join!(read("first", 0.1), read("second", 0.9));
        assert_eq!(first, (Some("first".into()), 0.1));
        assert_eq!(second, (Some("second".into()), 0.9));
    }
}
