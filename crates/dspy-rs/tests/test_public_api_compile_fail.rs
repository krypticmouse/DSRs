use std::fs;
use std::path::Path;
use std::process::Command;

fn run_compile_fail_case(name: &str, source: &str) -> String {
    let temp = tempfile::tempdir().expect("tempdir should be creatable");
    let case_dir = temp.path().join(name);
    fs::create_dir_all(case_dir.join("src")).expect("case src dir should be creatable");

    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ndspy-rs = {{ path = \"{}\" }}\nanyhow = \"1\"\n",
        manifest_path.display()
    );

    fs::write(case_dir.join("Cargo.toml"), cargo_toml).expect("cargo manifest should be writable");
    fs::write(case_dir.join("src/main.rs"), source).expect("source file should be writable");

    let output = Command::new("cargo")
        .arg("check")
        .arg("--quiet")
        .current_dir(&case_dir)
        .output()
        .expect("cargo check should run");

    assert!(
        !output.status.success(),
        "expected compile failure, but case compiled successfully:\n{}",
        source
    );

    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn dyn_predictor_is_not_publicly_importable() {
    let stderr = run_compile_fail_case(
        "private_dyn_predictor_case",
        r#"
use dspy_rs::DynPredictor;

fn main() {
    let _ = std::any::type_name::<Option<&'static dyn DynPredictor>>();
}
"#,
    );

    assert!(
        stderr.contains("DynPredictor")
            && (stderr.contains("private") || stderr.contains("no `DynPredictor` in the root")),
        "expected DynPredictor import failure, got:\n{stderr}"
    );
}

#[test]
fn named_parameters_is_not_publicly_importable() {
    let stderr = run_compile_fail_case(
        "private_named_parameters_case",
        r#"
use dspy_rs::named_parameters;

fn main() {
    let _ = named_parameters;
}
"#,
    );

    assert!(
        stderr.contains("named_parameters")
            && (stderr.contains("private") || stderr.contains("no `named_parameters` in the root")),
        "expected named_parameters import failure, got:\n{stderr}"
    );
}

#[test]
fn optimizer_compile_rejects_wrong_signature_input_type() {
    let stderr = run_compile_fail_case(
        "wrong_signature_case",
        r#"
use anyhow::Result;
use dspy_rs::{COPRO, ChainOfThought, Example, MetricOutcome, Optimizer, Predicted, Signature, TypedMetric, WithReasoning};

#[derive(Signature, Clone, Debug)]
struct RightSig {
    #[input]
    prompt: String,
    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct WrongSig {
    #[input]
    prompt_id: i64,
    #[output]
    answer: String,
}

struct Metric;

impl TypedMetric<RightSig, ChainOfThought<RightSig>> for Metric {
    async fn evaluate(
        &self,
        _example: &Example<RightSig>,
        _prediction: &Predicted<WithReasoning<RightSigOutput>>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(1.0))
    }
}

fn main() {
    let mut module = ChainOfThought::<RightSig>::new();
    let trainset: Vec<Example<WrongSig>> = Vec::new();
    let optimizer = COPRO::builder().breadth(1).depth(1).build();
    let _future = optimizer.compile::<WrongSig, _, _>(&mut module, trainset, &Metric);
}
"#,
    );

    assert!(
        stderr.contains("Module<Input = S::Input>")
            || stderr.contains("type mismatch")
            || stderr.contains("TypedMetric<WrongSig"),
        "expected optimizer signature mismatch failure, got:\n{stderr}"
    );
}
