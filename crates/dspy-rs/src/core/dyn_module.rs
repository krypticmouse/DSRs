use crate::{BamlValue, PredictError, Predicted, SignatureSchema};

use super::DynPredictor;

pub type StrategyConfig = serde_json::Value;
pub type StrategyConfigSchema = serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum StrategyError {
    #[error("unknown strategy `{name}`")]
    UnknownStrategy { name: String },
    #[error("duplicate strategy registration `{name}`")]
    DuplicateStrategy { name: &'static str },
    #[error("invalid config for strategy `{strategy}`: {reason}")]
    InvalidConfig {
        strategy: &'static str,
        reason: String,
    },
    #[error("failed to build strategy `{strategy}`: {reason}")]
    BuildFailed {
        strategy: &'static str,
        reason: String,
    },
}

#[async_trait::async_trait]
pub trait DynModule: Send + Sync {
    fn schema(&self) -> &SignatureSchema;
    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)>;
    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)>;
    async fn forward(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError>;
}

pub trait StrategyFactory: Send + Sync {
    fn name(&self) -> &'static str;
    fn config_schema(&self) -> StrategyConfigSchema;
    fn create(
        &self,
        base_schema: &SignatureSchema,
        config: StrategyConfig,
    ) -> std::result::Result<Box<dyn DynModule>, StrategyError>;
}

pub struct StrategyFactoryRegistration {
    pub factory: &'static dyn StrategyFactory,
}

inventory::collect!(StrategyFactoryRegistration);

pub mod registry {
    use std::collections::HashSet;

    use crate::SignatureSchema;

    use super::{DynModule, StrategyConfig, StrategyError, StrategyFactory};

    pub fn get(name: &str) -> std::result::Result<&'static dyn StrategyFactory, StrategyError> {
        let mut matches = inventory::iter::<super::StrategyFactoryRegistration>
            .into_iter()
            .filter(|registration| registration.factory.name() == name)
            .map(|registration| registration.factory);

        let first = matches
            .next()
            .ok_or_else(|| StrategyError::UnknownStrategy {
                name: name.to_string(),
            })?;

        if matches.next().is_some() {
            return Err(StrategyError::DuplicateStrategy { name: first.name() });
        }

        Ok(first)
    }

    pub fn create(
        name: &str,
        schema: &SignatureSchema,
        config: StrategyConfig,
    ) -> std::result::Result<Box<dyn DynModule>, StrategyError> {
        let factory = get(name)?;
        factory.create(schema, config)
    }

    pub fn list() -> Vec<&'static str> {
        let mut seen = HashSet::new();
        let mut names = Vec::new();
        for registration in inventory::iter::<super::StrategyFactoryRegistration> {
            let name = registration.factory.name();
            if seen.insert(name) {
                names.push(name);
            }
        }
        names.sort_unstable();
        names
    }
}
