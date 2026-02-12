use anyhow::{Result, anyhow};
use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use bon::Builder;
use dspy_rs::{
    COPRO, CallMetadata, DataLoader, Example, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedLoadOptions, TypedMetric, UnknownFieldPolicy,
    average_score, evaluate_trainset,
};
use facet;
use parquet::arrow::ArrowWriter;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

#[derive(Signature, Clone, Debug)]
struct LoaderSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct NumericSig {
    #[input]
    value: i64,

    #[output]
    doubled: i64,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct EchoModule {
    #[builder(default = Predict::<LoaderSig>::builder().instruction("seed").build())]
    predictor: Predict<LoaderSig>,
}

impl Module for EchoModule {
    type Input = LoaderSigInput;
    type Output = LoaderSigOutput;

    async fn forward(&self, input: LoaderSigInput) -> Result<Predicted<LoaderSigOutput>, PredictError> {
        let _ = &self.predictor;
        Ok(Predicted::new(
            LoaderSigOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

struct ExactMatch;

impl TypedMetric<LoaderSig, EchoModule> for ExactMatch {
    async fn evaluate(
        &self,
        example: &Example<LoaderSig>,
        prediction: &Predicted<LoaderSigOutput>,
    ) -> Result<MetricOutcome> {
        let score = (example.output.answer == prediction.answer) as u8 as f32;
        Ok(MetricOutcome::score(score))
    }
}

fn write_file(path: &Path, contents: &str) -> Result<()> {
    fs::write(path, contents)?;
    Ok(())
}

fn write_qa_parquet(path: &Path, questions: &[&str], answers: &[&str]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("question", DataType::Utf8, false),
        Field::new("answer", DataType::Utf8, false),
    ]));

    let question_col: ArrayRef = Arc::new(StringArray::from(questions.to_vec()));
    let answer_col: ArrayRef = Arc::new(StringArray::from(answers.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![question_col, answer_col])?;

    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_numeric_parquet(path: &Path, values: &[i64], doubled: &[i64]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("doubled", DataType::Int64, false),
    ]));

    let value_col: ArrayRef = Arc::new(Int64Array::from(values.to_vec()));
    let doubled_col: ArrayRef = Arc::new(Int64Array::from(doubled.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![value_col, doubled_col])?;

    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[test]
fn csv_typed_success_path() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(
        &path,
        "question,answer\nWhat is 2+2?,4\nCapital of France?,Paris\n",
    )?;

    let examples = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[0].input.question, "What is 2+2?");
    assert_eq!(examples[0].output.answer, "4");
    Ok(())
}

#[test]
fn csv_unknown_extra_columns_ignored_by_default() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(
        &path,
        "question,answer,notes\nWhat is 2+2?,4,math\nCapital of France?,Paris,geo\n",
    )?;

    let examples = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 2);
    Ok(())
}

#[test]
fn csv_unknown_columns_error_when_policy_is_error() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(
        &path,
        "question,answer,notes\nWhat is 2+2?,4,math\n",
    )?;

    let err = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions {
            field_map: HashMap::new(),
            unknown_fields: UnknownFieldPolicy::Error,
        },
    )
    .expect_err("unknown field policy should fail when extra columns exist");

    assert!(err.to_string().contains("unknown field `notes`"));
    Ok(())
}

#[test]
fn csv_missing_required_input_field_errors() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(&path, "answer\n4\n")?;

    let err = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
    )
    .expect_err("missing question field should fail");

    assert!(err.to_string().contains("missing field `question`"));
    Ok(())
}

#[test]
fn csv_missing_required_output_field_errors() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(&path, "question\nWhat is 2+2?\n")?;

    let err = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
    )
    .expect_err("missing answer field should fail");

    assert!(err.to_string().contains("missing field `answer`"));
    Ok(())
}

#[test]
fn csv_mapper_overload_success() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(&path, "q,a\nWhat is 2+2?,4\n")?;

    let examples = DataLoader::load_csv_with::<LoaderSig, _>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
        |row| {
            Ok(Example::new(
                LoaderSigInput {
                    question: row.get::<String>("q")?,
                },
                LoaderSigOutput {
                    answer: row.get::<String>("a")?,
                },
            ))
        },
    )?;

    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0].input.question, "What is 2+2?");
    assert_eq!(examples[0].output.answer, "4");
    Ok(())
}

#[test]
fn csv_mapper_overload_error_includes_row_index() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(&path, "q,a\nWhat is 2+2?,4\n")?;

    let err = DataLoader::load_csv_with::<LoaderSig, _>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
        |_row| Err(anyhow!("custom mapper failure")),
    )
    .expect_err("mapper failure should surface as row-indexed error");

    assert!(err.to_string().contains("mapper error at row 1"));
    assert!(err.to_string().contains("custom mapper failure"));
    Ok(())
}

#[test]
fn json_array_typed_success() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.json");
    write_file(
        &path,
        r#"[{"question":"What is 2+2?","answer":"4"},{"question":"Capital of France?","answer":"Paris"}]"#,
    )?;

    let examples = DataLoader::load_json::<LoaderSig>(
        path.to_str().unwrap(),
        false,
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[1].output.answer, "Paris");
    Ok(())
}

#[test]
fn json_mapper_overload_success() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.json");
    write_file(
        &path,
        r#"[{"prompt":"What is 2+2?","gold":"4"},{"prompt":"Capital of France?","gold":"Paris"}]"#,
    )?;

    let examples = DataLoader::load_json_with::<LoaderSig, _>(
        path.to_str().unwrap(),
        false,
        TypedLoadOptions::default(),
        |row| {
            Ok(Example::new(
                LoaderSigInput {
                    question: row.get::<String>("prompt")?,
                },
                LoaderSigOutput {
                    answer: row.get::<String>("gold")?,
                },
            ))
        },
    )?;

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[0].input.question, "What is 2+2?");
    assert_eq!(examples[1].output.answer, "Paris");
    Ok(())
}

#[test]
fn json_mapper_overload_error_includes_row_index() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.json");
    write_file(
        &path,
        r#"[{"question":"What is 2+2?","answer":"4"}]"#,
    )?;

    let err = DataLoader::load_json_with::<LoaderSig, _>(
        path.to_str().unwrap(),
        false,
        TypedLoadOptions::default(),
        |_row| Err(anyhow!("json mapper failed")),
    )
    .expect_err("mapper failure should surface as row-indexed error");

    assert!(err.to_string().contains("mapper error at row 1"));
    assert!(err.to_string().contains("json mapper failed"));
    Ok(())
}

#[test]
fn jsonl_typed_success() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.jsonl");
    write_file(
        &path,
        r#"{"question":"What is 2+2?","answer":"4"}
{"question":"Capital of France?","answer":"Paris"}
"#,
    )?;

    let examples = DataLoader::load_json::<LoaderSig>(
        path.to_str().unwrap(),
        true,
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[0].input.question, "What is 2+2?");
    Ok(())
}

#[test]
fn json_type_mismatch_errors() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("bad.json");
    write_file(
        &path,
        r#"[{"value":"not-an-int","doubled":2}]"#,
    )?;

    let err = DataLoader::load_json::<NumericSig>(
        path.to_str().unwrap(),
        false,
        TypedLoadOptions::default(),
    )
    .expect_err("invalid numeric input should fail conversion");

    assert!(err.to_string().contains("type mismatch"));
    Ok(())
}

#[test]
fn jsonl_type_mismatch_errors() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("bad.jsonl");
    write_file(
        &path,
        r#"{"value":1,"doubled":"not-an-int"}
"#,
    )?;

    let err = DataLoader::load_json::<NumericSig>(
        path.to_str().unwrap(),
        true,
        TypedLoadOptions::default(),
    )
    .expect_err("invalid numeric output should fail conversion");

    assert!(err.to_string().contains("type mismatch"));
    Ok(())
}

#[test]
fn parquet_typed_success_path() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.parquet");
    write_qa_parquet(
        &path,
        &["What is 2+2?", "Capital of France?"],
        &["4", "Paris"],
    )?;

    let examples = DataLoader::load_parquet::<LoaderSig>(
        path.to_str().unwrap(),
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[1].output.answer, "Paris");
    Ok(())
}

#[test]
fn parquet_mapper_overload_success() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.parquet");
    write_qa_parquet(&path, &["Q1"], &["A1"])?;

    let examples = DataLoader::load_parquet_with::<LoaderSig, _>(
        path.to_str().unwrap(),
        TypedLoadOptions::default(),
        |row| {
            Ok(Example::new(
                LoaderSigInput {
                    question: row.get::<String>("question")?,
                },
                LoaderSigOutput {
                    answer: row.get::<String>("answer")?,
                },
            ))
        },
    )?;

    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0].input.question, "Q1");
    assert_eq!(examples[0].output.answer, "A1");
    Ok(())
}

#[test]
fn hf_typed_from_parquet_success_path() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.parquet");
    write_qa_parquet(&path, &["Q1", "Q2"], &["A1", "A2"])?;

    let examples = DataLoader::load_hf_from_parquet::<LoaderSig>(
        vec![PathBuf::from(&path)],
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[0].output.answer, "A1");
    Ok(())
}

#[test]
fn typed_loader_field_remap_supports_input_and_output() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(&path, "prompt,completion\nWhat is 2+2?,4\n")?;

    let mut field_map = HashMap::new();
    field_map.insert("question".to_string(), "prompt".to_string());
    field_map.insert("answer".to_string(), "completion".to_string());

    let examples = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions {
            field_map,
            unknown_fields: UnknownFieldPolicy::Ignore,
        },
    )?;

    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0].input.question, "What is 2+2?");
    assert_eq!(examples[0].output.answer, "4");
    Ok(())
}

#[test]
fn parquet_numeric_round_trip_for_typed_conversion() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("numeric.parquet");
    write_numeric_parquet(&path, &[1, 2, 3], &[2, 4, 6])?;

    let examples = DataLoader::load_parquet::<NumericSig>(
        path.to_str().unwrap(),
        TypedLoadOptions::default(),
    )?;

    assert_eq!(examples.len(), 3);
    assert_eq!(examples[2].output.doubled, 6);
    Ok(())
}

#[tokio::test]
async fn typed_loader_outputs_feed_evaluator_and_optimizer_paths() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("train.csv");
    write_file(
        &path,
        "question,answer\none,one\ntwo,two\n",
    )?;

    let trainset = DataLoader::load_csv::<LoaderSig>(
        path.to_str().unwrap(),
        ',',
        true,
        TypedLoadOptions::default(),
    )?;

    let metric = ExactMatch;
    let mut module = EchoModule::builder().build();

    let outcomes = evaluate_trainset(&module, &trainset, &metric).await?;
    assert_eq!(outcomes.len(), 2);
    assert_eq!(average_score(&outcomes), 1.0);

    let optimizer = COPRO::builder().breadth(2).depth(1).build();
    optimizer
        .compile::<LoaderSig, _, _>(&mut module, trainset, &metric)
        .await?;

    Ok(())
}
