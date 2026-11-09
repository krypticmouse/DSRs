use anyhow::{Result, anyhow, ensure};

use super::{Engine, LeafInfo};
use crate::{LM, Predict, Signature};

/// Propose task-specific instructions, preserving the task's requirements.
/// Use the training observations and previous scored instructions to address
/// failures. Return distinct, complete replacement instructions, not advice
/// about how someone else could write an instruction.
#[derive(Signature, Clone, Debug)]
struct ProposeInstructions {
    #[input]
    task: String,
    #[input]
    current_instruction: String,
    #[input]
    observations: String,
    #[input]
    count: u32,
    #[output]
    instructions: Vec<String>,
}

pub(super) async fn propose(
    model: Option<&LM>,
    temperature: f32,
    leaf: &LeafInfo,
    current: &str,
    observations: String,
    count: usize,
    engine: &mut Engine,
) -> Result<Vec<String>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        engine.budget_allows(1),
        "budget exhausted before instruction proposal"
    );
    let mut lm = match model {
        Some(lm) => lm.clone(),
        None => crate::core::settings::GLOBAL_SETTINGS
            .read()
            .map_err(|_| anyhow!("settings lock poisoned"))?
            .as_ref()
            .map(|s| s.lm.as_ref().clone())
            .ok_or_else(|| {
                anyhow!("instruction proposal requires prompt_model or a configured global LM")
            })?,
    };
    lm.config.temperature = temperature;
    let predictor = Predict::<ProposeInstructions>::builder().lm(lm).build();
    engine.charge(0, 1);
    let result = predictor
        .call(ProposeInstructionsInput {
            task: leaf.schema_for_reflection(),
            current_instruction: current.to_owned(),
            observations,
            count: count.try_into()?,
        })
        .await?;
    let mut instructions = Vec::new();
    for instruction in &result.instructions {
        let instruction = instruction.trim();
        if !instruction.is_empty()
            && instruction != current
            && !instructions.iter().any(|i| i == instruction)
        {
            instructions.push(instruction.to_owned());
        }
        if instructions.len() == count {
            break;
        }
    }
    ensure!(
        !instructions.is_empty(),
        "proposal model returned no distinct nonempty instructions"
    );
    Ok(instructions)
}
