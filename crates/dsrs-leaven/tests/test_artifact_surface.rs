use std::ops::ControlFlow;

use anyhow::Result;
use dsrs_core::{
    CallMetadata, Facet, Module, PredictError, Predicted, Signature, visit_named_predictors_mut,
};
use dsrs_leaven::{
    DsrsModuleFactory, DsrsPredictorEdit, DsrsProgramArtifact, DsrsProgramChange,
    DsrsProgramSurface, PredictorPath,
};
use dsrs_predict::Predict;
use leaven_core::Artifact;
use leaven_surface::EditSurface;

#[derive(Signature, Clone, Debug)]
struct BridgeSig {
    /// Question to answer.
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Facet)]
struct BridgeProgram {
    predictor: Predict<BridgeSig>,
}

impl BridgeProgram {
    fn with_instruction(instruction: &str) -> Self {
        Self {
            predictor: Predict::<BridgeSig>::builder()
                .instruction(instruction)
                .build(),
        }
    }
}

impl Default for BridgeProgram {
    fn default() -> Self {
        Self::with_instruction("default bridge instruction")
    }
}

impl Module for BridgeProgram {
    type Input = BridgeSigInput;
    type Output = BridgeSigOutput;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        Ok(Predicted::new(
            BridgeSigOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

#[derive(Clone)]
struct BridgeFactory;

impl DsrsModuleFactory<BridgeProgram> for BridgeFactory {
    fn fresh_module(&self) -> BridgeProgram {
        BridgeProgram::default()
    }
}

fn predictor_instruction(program: &mut BridgeProgram) -> String {
    let mut instruction = None;
    visit_named_predictors_mut(program, |path, predictor| {
        if path == "predictor" {
            instruction = Some(predictor.instruction());
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .expect("predictor discovery should succeed");
    instruction.expect("test program has predictor")
}

#[test]
fn artifact_surface_applies_instruction_edits_immutably() -> Result<()> {
    let mut seed_program = BridgeProgram::with_instruction("solve carefully");
    let artifact = DsrsProgramArtifact::<BridgeSig, BridgeProgram, BridgeFactory>::capture(
        BridgeFactory,
        &mut seed_program,
    )?;
    let surface = DsrsProgramSurface::default();

    let parts = surface.parts(&artifact)?;
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].id, PredictorPath::new("predictor")?);
    assert_eq!(parts[0].address, PredictorPath::new("predictor")?);
    assert_eq!(parts[0].view, "solve carefully");

    let change = surface.change_part(
        &artifact,
        PredictorPath::new("predictor")?,
        "answer with an integer only".to_string(),
    )?;
    assert_eq!(change.path(), &PredictorPath::new("predictor")?);
    assert_eq!(
        change.edit(),
        &DsrsPredictorEdit::ReplaceInstruction("answer with an integer only".to_string())
    );

    let changed = artifact.apply_change(&change)?;
    assert_ne!(artifact.identity(), changed.identity());
    assert!(changed.cache_identity().is_some());

    let original_parts = surface.parts(&artifact)?;
    let changed_parts = surface.parts(&changed)?;
    assert_eq!(original_parts[0].view, "solve carefully");
    assert_eq!(changed_parts[0].view, "answer with an integer only");

    let mut materialized = changed.materialize_module()?;
    assert_eq!(
        predictor_instruction(&mut materialized),
        "answer with an integer only"
    );

    let direct_change = DsrsProgramChange::replace_instruction(
        PredictorPath::new("predictor")?,
        "use final-answer form".to_string(),
    );
    let changed_again = changed.apply_change(&direct_change)?;
    assert_eq!(
        surface.parts(&changed_again)?[0].view,
        "use final-answer form"
    );

    Ok(())
}
