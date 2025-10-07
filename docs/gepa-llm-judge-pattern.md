# Using LLM-as-a-Judge with GEPA

This guide explains how to use an LLM judge to automatically generate rich textual feedback for GEPA optimization, making it easier to optimize complex tasks where manual feedback rules are hard to codify.

## The Pattern

Instead of manually writing feedback rules, use another LLM to evaluate the output and reasoning:

```
Task LM → generates answer + reasoning
    ↓
Judge LM → analyzes quality and provides feedback
    ↓
GEPA Reflection LM → reads feedback and improves prompt
    ↓
Better Task LM prompt
```

## Why Use LLM-as-a-Judge?

**Good for:**
- Subjective quality assessment (writing style, helpfulness, clarity)
- Complex reasoning evaluation (is the logic sound?)
- Tasks where rules are hard to codify
- Analyzing reasoning quality beyond just answer correctness

**When to avoid:**
- You have deterministic checks (unit tests, schema validation)
- Verifiable correctness (code compilation, exact matches)
- Need fast/cheap evaluation
- Simple binary pass/fail

## Complete Example Walkthrough

See `examples/10-gepa-llm-judge.rs` for a full working example. Here are the key pieces:

### 1. Task Signature with Reasoning

```rust
#[Signature(cot)]
struct MathWordProblem {
    #[input]
    pub problem: String,
    
    #[output]
    pub reasoning: String,  // We want to optimize this too
    
    #[output]
    pub answer: String,
}
```

### 2. Judge Signature

```rust
#[Signature]
struct MathJudge {
    /// You are an expert math teacher evaluating student work.
    
    #[input]
    pub problem: String,
    
    #[input]
    pub expected_answer: String,
    
    #[input]
    pub student_answer: String,
    
    #[input]
    pub student_reasoning: String,
    
    #[output(desc = "Detailed evaluation of the work")]
    pub evaluation: String,  // This becomes the feedback
}
```

### 3. Module with Embedded Judge

```rust
#[derive(Builder, Optimizable)]
struct MathSolver {
    #[parameter]
    solver: Predict,  // This gets optimized
    
    judge: Predict,   // This stays fixed, just evaluates
    judge_lm: Arc<Mutex<LM>>,
}
```

### 4. FeedbackEvaluator with Judge

```rust
impl FeedbackEvaluator for MathSolver {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) 
        -> FeedbackMetric 
    {
        // Extract outputs
        let student_answer = prediction.get("answer", None).as_str().unwrap();
        let student_reasoning = prediction.get("reasoning", None).as_str().unwrap();
        let expected = example.get("expected_answer", None).as_str().unwrap();
        
        // Call the judge
        let judge_input = example! {
            "problem": "input" => problem,
            "expected_answer": "input" => expected,
            "student_answer": "input" => student_answer,
            "student_reasoning": "input" => student_reasoning
        };
        
        let judge_output = match self.judge
            .forward_with_config(judge_input, Arc::clone(&self.judge_lm))
            .await 
        {
            Ok(output) => output,
            Err(_) => {
                // Fallback if judge fails
                return FeedbackMetric::new(
                    if student_answer == expected { 1.0 } else { 0.0 },
                    format!("Expected: {}, Got: {}", expected, student_answer)
                );
            }
        };
        
        let judge_evaluation = judge_output
            .get("evaluation", None)
            .as_str()
            .unwrap_or("No evaluation provided")
            .to_string();
        
        // Score based on both correctness AND reasoning quality
        let answer_correct = student_answer.trim() == expected.trim();
        let good_reasoning = judge_evaluation.to_lowercase().contains("sound reasoning") 
            || judge_evaluation.to_lowercase().contains("correct approach");
        
        let score = match (answer_correct, good_reasoning) {
            (true, true) => 1.0,   // Perfect
            (true, false) => 0.7,  // Right answer, flawed reasoning
            (false, true) => 0.3,  // Wrong answer, but valid approach
            (false, false) => 0.0, // Completely wrong
        };
        
        // Combine factual info with judge's analysis
        let feedback = format!(
            "Problem: {}\nExpected: {}\nPredicted: {}\n\
             Answer: {}\n\nReasoning Quality Analysis:\n{}",
            problem, expected, student_answer,
            if answer_correct { "CORRECT" } else { "INCORRECT" },
            judge_evaluation
        );
        
        FeedbackMetric::new(score, feedback)
    }
}
```

## Key Benefits

**1. Catches lucky guesses:**
```
Answer: Correct (1.0)
But reasoning: "I just multiplied random numbers"
Score: 0.7 (penalized for bad reasoning)
```

**2. Rewards partial progress:**
```
Answer: Wrong
But reasoning: "Correct approach, arithmetic error in final step"
Score: 0.3 (partial credit)
```

**3. Identifies systematic issues:**
The judge notices patterns like:
- "Model consistently skips showing intermediate steps"
- "Model confuses similar concepts (area vs perimeter)"
- "Model doesn't check units in answers"

GEPA's reflection can then say:
> "Add explicit instruction to show all intermediate steps and verify units"

## Cost Considerations

LLM judges double your evaluation cost since every prediction requires:
1. Task LM call (generate answer)
2. Judge LM call (evaluate quality)

**Budget accordingly:**

```rust
GEPA::builder()
    .num_iterations(3)           // Fewer iterations
    .minibatch_size(3)           // Smaller batches
    .maybe_max_lm_calls(Some(100))  // Explicit limit
    .build()
```

**Optimization tips:**
- Use a cheaper model for judging (gpt-4o-mini vs gpt-4)
- Judge only failed examples (not ones that passed)
- Cache judge evaluations for identical outputs
- Use parallel evaluation to reduce wall-clock time

## When to Use Hybrid Approach

Best results often come from combining explicit checks with LLM judging:

```rust
async fn feedback_metric(&self, example: &Example, prediction: &Prediction) 
    -> FeedbackMetric 
{
    let mut feedback_parts = vec![];
    let mut score = 1.0;
    
    // Explicit checks first (fast, cheap, deterministic)
    if !is_valid_json(output) {
        feedback_parts.push("Invalid JSON format");
        score = 0.0;
    }
    
    if missing_required_fields(output) {
        feedback_parts.push("Missing fields: user_id, timestamp");
        score *= 0.5;
    }
    
    // Only call judge if basic checks pass
    if score > 0.0 {
        let judge_feedback = self.judge_quality(example, prediction).await;
        feedback_parts.push(judge_feedback);
        
        if judge_feedback.contains("low quality") {
            score *= 0.7;
        }
    }
    
    FeedbackMetric::new(score, feedback_parts.join("\n"))
}
```

## Example Output

When you run the example, GEPA will evolve prompts based on judge feedback:

```
Baseline: "Solve the math word problem step by step"
  → Some solutions skip steps
  → Judge: "Reasoning incomplete, jumped from step 2 to answer"

After GEPA:
  → "Solve step by step. Show ALL intermediate calculations. Label each step clearly."
  → Judge: "Sound reasoning, all steps shown clearly"
```

The judge's analysis becomes the signal that drives prompt improvement.

## Running the Example

```bash
OPENAI_API_KEY=your_key cargo run --example 10-gepa-llm-judge
```

This will show:
1. Baseline performance
2. Judge evaluations during optimization
3. How feedback evolves the prompt
4. Final test with judge analysis
