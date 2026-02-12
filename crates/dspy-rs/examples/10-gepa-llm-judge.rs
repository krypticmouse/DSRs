/*
Example: GEPA optimization with an LLM-as-a-judge typed metric.

Run with:
```
OPENAI_API_KEY=your_key cargo run --example 10-gepa-llm-judge
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    ChatAdapter, Example, FeedbackMetric, GEPA, LM, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric, average_score, configure, evaluate_trainset,
    init_tracing,
};

#[derive(Signature, Clone, Debug)]
struct MathWordProblem {
    /// Solve the problem step by step.

    #[input]
    problem: String,

    #[output]
    reasoning: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct MathJudge {
    /// Evaluate student reasoning and answer quality.

    #[input(desc = "The original problem")]
    problem: String,

    #[input(desc = "Expected answer")]
    expected_answer: String,

    #[input(desc = "Student answer")]
    student_answer: String,

    #[input(desc = "Student reasoning")]
    student_reasoning: String,

    #[output(desc = "Evaluation of the solution quality")]
    evaluation: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct MathSolver {
    #[builder(default = Predict::<MathWordProblem>::new())]
    solver: Predict<MathWordProblem>,
}

impl Module for MathSolver {
    type Input = MathWordProblemInput;
    type Output = MathWordProblemOutput;

    async fn forward(
        &self,
        input: MathWordProblemInput,
    ) -> Result<Predicted<MathWordProblemOutput>, PredictError> {
        self.solver.call(input).await
    }
}

struct LlmJudgeMetric {
    judge: Predict<MathJudge>,
}

impl TypedMetric<MathWordProblem, MathSolver> for LlmJudgeMetric {
    async fn evaluate(
        &self,
        example: &Example<MathWordProblem>,
        prediction: &Predicted<MathWordProblemOutput>,
    ) -> Result<MetricOutcome> {
        let problem = example.input.problem.clone();
        let expected = example.output.answer.clone();

        let student_answer = prediction.answer.clone();
        let student_reasoning = prediction.reasoning.clone();
        let exact_match = student_answer.trim() == expected.trim();

        let judge_output = self
            .judge
            .call(MathJudgeInput {
                problem: problem.clone(),
                expected_answer: expected.clone(),
                student_answer: student_answer.clone(),
                student_reasoning: student_reasoning.clone(),
            })
            .await;

        let (score, evaluation_text) = match judge_output {
            Ok(evaluation) => {
                let evaluation_text = evaluation.evaluation.clone();
                let score = if exact_match {
                    if evaluation_text.to_lowercase().contains("clear")
                        || evaluation_text.to_lowercase().contains("correct")
                    {
                        1.0
                    } else {
                        0.7
                    }
                } else if evaluation_text.to_lowercase().contains("partially")
                    || evaluation_text.to_lowercase().contains("good start")
                {
                    0.3
                } else {
                    0.0
                };
                (score, evaluation_text)
            }
            Err(err) => {
                let fallback = format!(
                    "judge call failed: {err}; expected={expected}; predicted={student_answer}"
                );
                ((exact_match as u8 as f32), fallback)
            }
        };

        let feedback = FeedbackMetric::new(
            score,
            format!(
                "problem={problem}\nexpected={expected}\npredicted={student_answer}\njudge={evaluation_text}"
            ),
        );

        Ok(MetricOutcome::with_feedback(score, feedback))
    }
}

fn training_example(problem: &str, expected_answer: &str) -> Example<MathWordProblem> {
    Example::new(
        MathWordProblemInput {
            problem: problem.to_string(),
        },
        MathWordProblemOutput {
            reasoning: String::new(),
            answer: expected_answer.to_string(),
        },
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(LM::builder().temperature(0.7).build().await?, ChatAdapter);

    let trainset = vec![
        training_example(
            "Sarah has 12 apples. She gives 3 away and buys 5 more. How many apples now?",
            "14",
        ),
        training_example(
            "A train travels 60 miles in 1 hour. How far in 3.5 hours?",
            "210",
        ),
        training_example(
            "There are 24 students. If 1/3 are absent, how many are present?",
            "16",
        ),
    ];

    let mut module = MathSolver::builder().build();
    let metric = LlmJudgeMetric {
        judge: Predict::<MathJudge>::builder()
            .instruction("Be strict and specific when grading student work.")
            .build(),
    };

    let baseline = average_score(&evaluate_trainset(&module, &trainset, &metric).await?);
    println!("Baseline score: {baseline:.3}");

    let gepa = GEPA::builder()
        .num_iterations(3)
        .minibatch_size(2)
        .temperature(0.9)
        .track_stats(true)
        .build();

    let result = gepa.compile(&mut module, trainset.clone(), &metric).await?;

    println!("Best score: {:.3}", result.best_candidate.average_score());
    println!("Total rollouts: {}", result.total_rollouts);
    println!("Total LM calls: {}", result.total_lm_calls);
    println!("Best instruction: {}", result.best_candidate.instruction);

    let test_problem =
        "A store sells pencils for $0.25 each. If you buy 8 pencils, what is the total?";
    let test_predicted = module
        .call(MathWordProblemInput {
            problem: test_problem.to_string(),
        })
        .await?;
    let test_example = training_example(test_problem, "2");
    let test_metric = metric.evaluate(&test_example, &test_predicted).await?;

    println!("Test answer: {}", test_predicted.answer);
    println!("Test score: {:.3}", test_metric.score);
    if let Some(feedback) = test_metric.feedback {
        println!("Judge feedback:\n{}", feedback.feedback);
    }

    Ok(())
}
