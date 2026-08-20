/*
Front Desk, Act I: the contract is the prompt.

The support-desk milestone from chapters 1-4 of the Learn book
(docs/docs/learn): typed signatures for the fictional Sourdough app.

This example shows:
1. Enums and nested structs as output types (`#[Schema]`)
2. `Option<T>` as formal permission to say "nothing to report"
3. Soft `#[check]` and hard `#[assert]` constraints (minijinja expressions —
   `this|length > 0`, never Rust method calls)
4. Demos + a per-predictor LM on the `Predict` builder
5. `ChainOfThought` as the same contract plus a `reasoning` field
6. The exact prompt a signature compiles into (`ChatAdapter`)

Run with:
```
cargo run --example 20-frontdesk-contract
```
*/

use anyhow::Result;
use dspy_rs::{
    ChainOfThought, ChatAdapter, Demo, LM, Predict, Schema, Signature, configure, init_tracing,
};

// ---------------------------------------------------------------------------
// The types do the talking
// ---------------------------------------------------------------------------

#[Schema]
#[derive(Clone, Debug)]
enum Category {
    Bug,
    Billing,
    HowTo,
    FeatureRequest,
}

#[Schema]
#[derive(Clone, Debug)]
enum Mood {
    Calm,
    Confused,
    Frustrated,
    Furious,
}

#[Schema]
#[derive(Clone, Debug)]
struct OrderRef {
    /// The order number exactly as the customer wrote it
    order_id: String,
    /// The sentence of the ticket where it appears
    quote: String,
}

// ---------------------------------------------------------------------------
// The contracts
// ---------------------------------------------------------------------------

/// Classify a support ticket for the Sourdough app.
#[derive(Signature, Clone, Debug)]
struct Triage {
    /// The full text of the customer's ticket
    #[input]
    ticket: String,

    #[output]
    category: Category,
}

/// Extract the facts a support agent needs from a Sourdough ticket.
/// Do not guess. If a fact is not in the ticket, leave it out.
#[derive(Signature, Clone, Debug)]
struct ExtractFacts {
    /// The full text of the customer's ticket
    #[input]
    ticket: String,

    #[output]
    category: Category,

    #[output]
    mood: Mood,

    /// Present only when the customer actually references an order
    #[output]
    order: Option<OrderRef>,

    /// One sentence a busy human reads instead of the ticket
    #[output]
    #[assert("this|length > 0")]
    summary: String,

    /// How confident you are in the category, 0 to 1
    #[output]
    #[check("this >= 0.0 and this <= 1.0", label = "confidence_range")]
    confidence: f64,
}

/// Write a warm, concrete reply to a Sourdough support ticket.
/// Never promise a refund. Sign off as "The Sourdough Team".
#[derive(Signature, Clone, Debug)]
struct DraftReply {
    /// The customer's ticket
    #[input]
    ticket: String,
    /// One-sentence summary from triage
    #[input]
    summary: String,
    #[input]
    mood: Mood,

    /// The reply to send
    #[output]
    reply: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // What did the model actually see? The adapter compiles the signature
    // into the prompt — print it instead of trusting anyone's word.
    let adapter = ChatAdapter;
    println!("=== the prompt Triage becomes ===\n");
    println!(
        "{}",
        adapter.build_system_def(
            dspy_rs::ir::SignatureDef::of::<Triage>(),
            dspy_rs::ir::SignatureDef::types_of::<Triage>(),
            None,
        )
    );

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
    );

    // Cheap pattern work and company-voice work should not share a model.
    let fast = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await?;
    let strong = LM::builder()
        .model("openai:gpt-4o".to_string())
        .temperature(0.7)
        .build()
        .await?;

    // Demos: typed input/output pairs that ride along as worked examples.
    // The signature stays untouched — the baggage lives in the caller.
    let triage = Predict::<Triage>::builder()
        .demo(Demo::<Triage>::new(
            TriageInput {
                ticket: "The update crashed mid-payment and I got charged twice.".into(),
            },
            TriageOutput {
                category: Category::Billing,
            },
        ))
        .demo(Demo::<Triage>::new(
            TriageInput {
                ticket: "App crashes every time I open the feeding log.".into(),
            },
            TriageOutput {
                category: Category::Bug,
            },
        ))
        .lm(fast)
        .build();

    let extract = Predict::<ExtractFacts>::new();
    let drafter = ChainOfThought::<DraftReply>::builder().lm(strong).build();

    let herman_ticket = "I updated last night and now my starter tracker shows 0 days. \
                         Did the update kill my starter data?? Herman is FOUR YEARS OLD."
        .to_string();

    println!("=== triage (2 demos, fast model) ===\n");
    let out = triage
        .call(TriageInput {
            ticket: herman_ticket.clone(),
        })
        .await?;
    println!("category: {:?}\n", out.category);

    println!("=== extract (five questions, one call) ===\n");
    let out = extract
        .call(ExtractFactsInput {
            ticket: herman_ticket.clone(),
        })
        .await?;

    println!(
        "{:?} | {:?} | conf {:.2}",
        out.category, out.mood, out.confidence
    );
    println!("summary: {}", out.summary);
    println!("order:   {:?}", out.order);

    // The soft constraint's verdict rides along with the result.
    if let Some(meta) = out.metadata().field_meta.get("confidence") {
        for check in &meta.checks {
            println!("check {}: passed = {}", check.label, check.passed);
        }
    }

    println!("\n=== draft (ChainOfThought, strong model) ===\n");
    let draft = drafter
        .call(DraftReplyInput {
            ticket: herman_ticket,
            summary: out.summary.clone(),
            mood: out.mood.clone(),
        })
        .await?;
    println!("reasoning: {}", draft.reasoning);
    println!("reply:\n{}", draft.reply);

    // Everything that is not an output field arrives via metadata.
    let usage = &draft.metadata().lm_usage;
    println!(
        "\n{} in / {} out",
        usage.prompt_tokens, usage.completion_tokens
    );

    Ok(())
}
