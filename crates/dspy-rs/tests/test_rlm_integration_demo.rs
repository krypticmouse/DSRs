#![cfg(feature = "rlm")]
#![allow(legacy_derive_helpers)]

use dspy_rs::modules::rlm::PyO3Runtime;
use dspy_rs::{
    ChatAdapter, LM, LMClient, Rlm, Signature, TestCompletionModel, configure, rlm_type,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;

use dspy_rs::__macro_support::pyo3;
use pyo3::types::{PyAnyMethods, PyDict};
use pyo3::{IntoPyObjectExt, Python};

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

async fn build_test_lm_with_client(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .temperature(0.0)
            .build(),
    )
    .await
    .expect("build lm")
    .with_client(LMClient::Test(client.clone()))
    .await
    .expect("install test client");
    (lm, client)
}

async fn configure_test_lm_with_client(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let (lm, client) = build_test_lm_with_client(responses).await;
    configure(lm.clone(), ChatAdapter::new());
    (lm, client)
}

#[rlm_type]
#[rlm(iter = "keywords", index = "keywords")]
#[derive(Clone, Debug)]
struct Paper {
    /// Paper title.
    title: String,
    /// Abstract body.
    abstract_text: String,
    /// Publication year.
    year: i32,
    /// Search keywords.
    keywords: Vec<String>,
    #[rlm(skip_python)]
    internal_rank: i32,
}

#[derive(Signature, Clone, Debug)]
/// Find the most relevant papers for the query.
struct PaperSearch {
    #[input]
    papers: Vec<Paper>,
    #[input]
    query: String,
    #[output]
    relevant_titles: Vec<String>,
    #[output]
    reasoning: String,
}

#[derive(Signature, Clone, Debug)]
/// Render-only signature for previewing a single paper object.
struct PaperPreviewSig {
    #[input]
    paper: Paper,
    #[output]
    ok: bool,
}

fn demo_papers() -> Vec<Paper> {
    vec![
        Paper {
            title: "Intro to Rust for LLMs".to_string(),
            abstract_text: "Typed pipelines for model programs.".to_string(),
            year: 2024,
            keywords: vec!["rust".to_string(), "llm".to_string()],
            internal_rank: 1,
        },
        Paper {
            title: "Graph Reasoning at Scale".to_string(),
            abstract_text: "Large context retrieval and synthesis.".to_string(),
            year: 2023,
            keywords: vec!["graph".to_string(), "retrieval".to_string()],
            internal_rank: 2,
        },
    ]
}

#[test]
fn demo_signature_generates_correct_pyclass_methods() {
    Python::attach(|py| {
        let paper = demo_papers().remove(0);
        let py_obj = paper
            .clone()
            .into_py_any(py)
            .expect("Paper should convert to native PyO3 object");
        let bound = py_obj.bind(py);

        assert!(
            !bound.is_instance_of::<PyDict>(),
            "Paper must inject as native object, not dict"
        );
        assert_eq!(
            bound
                .getattr("title")
                .expect("title getter")
                .extract::<String>()
                .expect("title extract"),
            paper.title
        );
        assert!(
            !bound
                .hasattr("internal_rank")
                .expect("hasattr internal_rank"),
            "skip_python fields must not be exposed as Python attributes"
        );

        let repr = bound
            .repr()
            .expect("repr")
            .extract::<String>()
            .expect("repr string");
        assert!(repr.contains("Paper"));

        let baml = bound.call_method0("__baml__").expect("__baml__ call");
        assert!(baml.is_instance_of::<PyDict>());
        let baml_dict = baml.cast::<PyDict>().expect("__baml__ returns dict");
        assert_eq!(
            baml_dict
                .get_item("title")
                .expect("title get_item")
                .extract::<String>()
                .expect("title value"),
            "Intro to Rust for LLMs"
        );
        assert_eq!(
            bound
                .call_method0("__len__")
                .expect("len call")
                .extract::<usize>()
                .expect("len value"),
            2
        );
    });
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn integration_preview_matches_spec_and_prompt_is_clean() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = configure_test_lm_with_client(vec![
        "dict_style_ok = True\ntry:\n    _ = papers[0]['title']\nexcept Exception:\n    dict_style_ok = False\nreason = f\"{type(papers[0]).__name__}:{papers[0].title}:dict={dict_style_ok}\"\nSUBMIT(relevant_titles=[papers[0].title], reasoning=reason)".to_string(),
    ])
    .await;

    let rlm = Rlm::<PaperSearch>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(1)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(PaperSearchInput {
            papers: demo_papers(),
            query: "rust typed model pipelines".to_string(),
        })
        .await
        .expect("RLM demo call should complete");

    assert_eq!(predicted.relevant_titles, vec!["Intro to Rust for LLMs"]);
    assert!(
        predicted
            .reasoning
            .contains("Paper:Intro to Rust for LLMs:dict=False"),
        "reasoning should confirm native object access and dict-style failure"
    );

    let request = client
        .last_request()
        .expect("expected action turn request capture");
    let request_debug = format!("{request:?}");

    assert!(
        request_debug.contains("## Task"),
        "system prompt should include Task section"
    );
    assert!(
        request_debug.contains("Find the most relevant papers for the query."),
        "system prompt should include developer instruction"
    );
    assert!(
        !request_debug.contains("Your input fields are:"),
        "adapter wrapping should be absent"
    );
    assert!(
        !request_debug.contains("Your objective is:"),
        "adapter wrapping should be absent"
    );

    assert!(
        request_debug.contains("## Input Variables"),
        "{request_debug}"
    );
    assert!(
        request_debug.contains("Variable: `papers` (access it in your code)"),
        "{request_debug}"
    );
    assert!(request_debug.contains("title: string"), "{request_debug}");
    assert!(
        request_debug.contains("=== Execution Receipt (Turn 1) ==="),
        "{request_debug}"
    );
    assert!(
        request_debug.contains("Budget: 1 turn remaining |"),
        "{request_debug}"
    );
    assert!(request_debug.contains("[query]"), "{request_debug}");
    assert!(
        request_debug.contains("=== Namespace ==="),
        "{request_debug}"
    );
    assert!(request_debug.contains("[Injected]"), "{request_debug}");
    assert!(request_debug.contains("[Recent]"), "{request_debug}");
    assert!(request_debug.contains(">>>"), "{request_debug}");
    assert!(
        !request_debug.contains("__baml__"),
        "preview should hide __baml__"
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn integration_preview_shows_paper_fields_and_methods() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = configure_test_lm_with_client(vec!["SUBMIT(ok=True)".to_string()]).await;

    let rlm = Rlm::<PaperPreviewSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(1)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(PaperPreviewSigInput {
            paper: demo_papers().remove(0),
        })
        .await
        .expect("preview demo should submit");
    assert!(predicted.ok);

    let request = client
        .last_request()
        .expect("expected preview request capture");
    let request_debug = format!("{request:?}");
    assert!(
        request_debug.contains("Variable: `paper` (access it in your code)"),
        "{request_debug}"
    );
    assert!(request_debug.contains("title: string"), "{request_debug}");
    assert!(
        request_debug.contains("## Input Variables"),
        "{request_debug}"
    );
    assert!(
        !request_debug.contains("Methods:"),
        "legacy methods block should not appear in new schema format"
    );
    assert!(
        !request_debug.contains(".__len__("),
        "dunder methods should not appear in schema-facing method surface"
    );
    assert!(
        !request_debug.contains("__baml__"),
        "preview should hide __baml__"
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn submit_validation_errors_are_pythonic_not_baml_internal() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = configure_test_lm_with_client(vec![
        "SUBMIT(relevant_titles=123, reasoning=5)".to_string(),
        "SUBMIT(relevant_titles=['Intro to Rust for LLMs'], reasoning='fixed')".to_string(),
    ])
    .await;

    let rlm = Rlm::<PaperSearch>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(2)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(PaperSearchInput {
            papers: demo_papers(),
            query: "rust".to_string(),
        })
        .await
        .expect("second submit should succeed after validation feedback");

    assert_eq!(predicted.relevant_titles, vec!["Intro to Rust for LLMs"]);
    assert_eq!(predicted.reasoning, "fixed");

    let request = client
        .last_request()
        .expect("expected second-turn request capture");
    let request_debug = format!("{request:?}");

    assert!(
        request_debug.contains("SubmitError: Validation failed"),
        "{request_debug}"
    );
    assert!(request_debug.contains("got python"), "{request_debug}");
    assert!(
        !request_debug.contains("BamlValue::"),
        "feedback should not leak Baml internal type names"
    );
}
