/*
Script to iterate and update the predictors of a module via the typed walker.

Run with:
```
cargo run --example 02-module-iteration-and-updation
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{Predict, Signature, init_tracing, named_parameters, named_parameters_ref};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct Rate {
    #[input]
    question: String,

    #[input]
    answer: String,

    #[output]
    rating: i8,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct QARater {
    #[builder(default = Predict::<QA>::builder().instruction("Answer clearly.").build())]
    answerer: Predict<QA>,

    #[builder(default = Predict::<Rate>::builder().instruction("Rate from 1 to 10.").build())]
    rater: Predict<Rate>,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct NestedModule {
    #[builder(default = QARater::builder().build())]
    qa_outer: QARater,

    #[builder(default = QARater::builder().build())]
    qa_inner: QARater,

    #[builder(default = Predict::<QA>::builder().instruction("Extra QA predictor.").build())]
    extra: Predict<QA>,
}

fn print_instructions<T>(label: &str, module: &T) -> Result<()>
where
    T: for<'a> facet::Facet<'a>,
{
    println!("{label}");
    let params = named_parameters_ref(module)?;
    for (path, predictor) in params {
        println!("  {path} -> {}", predictor.instruction());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    let mut qa_rater = QARater::builder().build();
    {
        let mut params = named_parameters(&mut qa_rater)?;
        for (path, predictor) in params.iter_mut() {
            predictor.set_instruction(format!("Updated instruction for `{path}`"));
        }
    }
    print_instructions("single module", &qa_rater)?;

    let mut nested = NestedModule::builder().build();
    {
        let mut params = named_parameters(&mut nested)?;
        for (path, predictor) in params.iter_mut() {
            predictor.set_instruction(format!("Deep updated: `{path}`"));
        }
    }
    print_instructions("nested module", &nested)?;

    Ok(())
}
