//! Typed predictors and prompting modules.

pub mod chain_of_thought;
pub mod predict;
pub mod react;

pub use chain_of_thought::{ChainOfThought, ChainOfThoughtOutput, Reasoning, WithReasoning};
pub use dsrs_core::*;
pub use dsrs_lm::*;
pub use predict::*;
pub use react::ReAct;
