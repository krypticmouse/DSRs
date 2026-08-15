/*
Front Desk, Act II (struct lane): pipelines are just structs.

The composition milestone from chapter 5 of the Learn book: two predictors in
a struct, a `Module` impl whose `forward` body is ordinary Rust, and the same
pipeline again in the functional (`fx`) lane with named call sites.

Run with:
```
cargo run --example 21-frontdesk-pipeline
```
*/

use anyhow::Result;
use dspy_rs::{
    ChainOfThought, LM, Module, Predict, PredictError, Predicted, Schema, Signature, WithReasoning,
    configure, fx, init_tracing,
};

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

    /// One sentence a busy human reads instead of the ticket
    #[output]
    summary: String,
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

// ---------------------------------------------------------------------------
// The struct lane: a pipeline is a struct with two fields and a method
// ---------------------------------------------------------------------------

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct FrontDesk {
    extract: Predict<ExtractFacts>,
    drafter: ChainOfThought<DraftReply>,
}

impl Module for FrontDesk {
    type Input = ExtractFactsInput;
    type Output = WithReasoning<DraftReplyOutput>;

    async fn forward(
        &self,
        input: ExtractFactsInput,
    ) -> Result<Predicted<Self::Output>, PredictError> {
        let ticket = input.ticket.clone();
        let facts = self.extract.call(input).await?;

        self.drafter
            .call(DraftReplyInput {
                ticket,
                summary: facts.summary.clone(),
                mood: facts.mood.clone(),
            })
            .await
    }
}

// ---------------------------------------------------------------------------
// The quick lane: fx calls with stable names ("extract", "drafter")
// ---------------------------------------------------------------------------

async fn front_desk(ticket: String) -> anyhow::Result<String> {
    let facts = fx::predict::<ExtractFacts>(
        "extract",
        ExtractFactsInput {
            ticket: ticket.clone(),
        },
    )
    .await?;

    let draft = fx::predict::<DraftReply>(
        "drafter",
        DraftReplyInput {
            ticket,
            summary: facts.summary.clone(),
            mood: facts.mood.clone(),
        },
    )
    .await?;

    Ok(draft.reply.clone())
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
    );

    let herman_ticket = "I updated last night and now my starter tracker shows 0 days. \
                         Did the update kill my starter data?? Herman is FOUR YEARS OLD."
        .to_string();

    println!("=== struct lane ===\n");
    let desk = FrontDesk {
        extract: Predict::new(),
        drafter: ChainOfThought::new(),
    };
    let out = desk
        .call(ExtractFactsInput {
            ticket: herman_ticket.clone(),
        })
        .await?;
    println!("{}", out.reply);

    println!("\n=== fx lane ===\n");
    let reply = front_desk(herman_ticket).await?;
    println!("{reply}");

    Ok(())
}
