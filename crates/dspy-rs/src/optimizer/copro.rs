use bon::Builder;
use crate::{Module, Example, Prediction, LM, Optimizer, Signature, Predict, MetaSignature, Predictor, Evaluator};
use anyhow::Result;
use validator::Validate;
use futures::future::join_all;
use std::collections::HashMap;


#[Signature]
struct BasicGenerateInstruction {
    #[input]
    pub basic_instruction: String,
    #[output]
    pub proposed_instruction: String,
}

#[Signature]
struct GenerateInstructionGivenAttempts {
    #[input]
    pub attempted_instruction: HashMap<String, f32>,
    #[output]
    pub proposed_instruction: String,
}

#[derive(Builder, Validate)]
struct COPRO {
    pub metric: Fn(&Example, &Prediction) -> f32,
    #[validate(range(min = 1))]
    #[builder(default = 10)]
    pub breadth: u8,
    #[validate(range(min = 1))]
    #[builder(default = 3)]
    pub depth: u8,
    #[builder(default = false)]
    pub track_stats: bool,
    pub prompt_model: Option<LM>,
}

impl COPRO {
    fn new(metric: Fn(&Example, &Prediction) -> f32) -> Self {
        Self::builder()
            .metric(metric);
    }
}

impl Optimizer for COPRO {
    const BASIC_GENERATOR: Predict = Predict::new(BasicGenerateInstruction::new());
    const REFINEMENT_GENERATOR: Predict = Predict::new(GenerateInstructionGivenAttempts::new());

    fn compile(&self, module: &Module, trainset: Vec<Example>) -> Result<()> {
        let evaluator = Evaluator::new(self.metric);
        let named_predictors = module.named_predictors();
        let init_evals = evaluator.evaluate(trainset.clone(), module).await?;

        let mut candidates: HashMap<String, Vec<String>> = HashMap::new();

        for (predictor_name, predictor) in named_predictors {
            let instruction = self.get_signature(predictor).instruction();
            let proposed_instructions = {
                let futures = vec![];

                for _ in 0..self.breadth {
                    if self.prompt_model.is_some() {
                        futures.push(BASIC_GENERATOR.forward_with_config(example! {
                            "basic_instruction": "input" => instruction
                        }, self.prompt_model.as_ref().unwrap()).await?)
                    } else {
                        futures.push(BASIC_GENERATOR.forward(example! {
                            "basic_instruction": "input" => instruction
                        }).await?)
                    }
                }

                join_all(futures).await
            };
            candidates.insert(predictor_name, proposed_instructions);
        }

        let curr_candidates = candidates;
        let new_module = module.clone();
        let new_named_predictors = new_module.named_predictors();

        let mut best_candidates = HashMap::new();
        for (predictor_name, predictor) in new_named_predictors {
            best_candidates.insert(predictor_name, predictor.signature.instruction());
        }

        let mut best_scores = 0.0;

        for d in 0..self.depth {
            println!(format!("Iteration Depth: {d}"));

            for (predictor_name, candidate_instructions) in curr_candidates.iter() {
                if new_named_predictors.get(predictor_name).FROZEN {
                    continue;
                }
                
                let mut candidates_evals = HashMap::new();

                for candidate_instruction in candidate_instructions {
                    new_named_predictors.get(predictor_name)?.signature.update_instruction(candidate_instruction);
                    let score = evaluator.evaluate(trainset.clone(), &new_module).await?;

                    if score > best_scores {
                        best_scores = score;
                        best_candidates.get_mut(predictor_name) = candidate_instruction;
                    }
                    
                    candidates_evals.insert(candidate_instruction, score);
                }
                
                let refined_instructions = REFINEMENT_GENERATOR.forward(example! {
                    "attempted_instruction": "input" => candidates_evals
                }).await?.get("proposed_instruction")?;
                let refined_scores = evaluator.evaluate(trainset.clone(), &new_module).await?;

                if refined_scores > best_scores {
                    best_scores = refined_scores;
                    best_candidates.get_mut(predictor_name) = refined_instructions;
                }
                
                curr_candidates.insert(predictor_name, refined_instructions);
            }
        }

        if self.track_stats {
            println!("Best scores: {}", best_scores);
            println!("Best candidates: {:?}", best_candidates);
        }

        module = new_module;
        Ok(())
    }
}