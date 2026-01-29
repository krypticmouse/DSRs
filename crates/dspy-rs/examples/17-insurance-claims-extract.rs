/*
Extract insurance claims with DSRs using the typed BAML schema.

Run with:
OPENAI_API_KEY=your_key cargo run --example 17-insurance-claims-extract -- \
  --input ../data/insurance_claims_extraction.parquet \
  --output ../data/structured_output_dsrs.json
*/

use anyhow::{bail, Context, Result};
use dspy_rs::{configure, ChatAdapter, DataLoader, LM, Predict, PredictError, Signature};
use dspy_rs::BamlType;
use serde::Serialize;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

// Keep the example self-contained; dates are represented as YYYY-MM-DD strings.
type NaiveDate = String;

/// Basic claim information (metadata about the claim intake).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
pub struct ClaimHeader {
    /// Claim ID in format `CLM-XXXXXX`, where `X` is a digit.
    pub claim_id: Option<String>,

    /// Date claim was reported in `YYYY-MM-DD` format.
    pub report_date: Option<NaiveDate>,

    /// Date incident occurred in `YYYY-MM-DD` format.
    pub incident_date: Option<NaiveDate>,

    /// Full name of person reporting claim.
    pub reported_by: Option<String>,

    /// Channel used to report claim.
    pub channel: Option<ClaimChannel>,
}

/// Channel used to report a claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
#[baml(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum ClaimChannel {
    Email,
    Phone,
    Portal,
    InPerson,
}

/// Policy information if available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
pub struct PolicyDetails {
    /// Policy number in format `POL-XXXXXXXXX`, where `X` is a digit.
    pub policy_number: Option<String>,

    /// Full legal name on policy.
    pub policyholder_name: Option<String>,

    /// Type of insurance coverage.
    pub coverage_type: Option<CoverageType>,

    /// Policy effective start date in `YYYY-MM-DD` format.
    pub effective_date: Option<NaiveDate>,

    /// Policy expiration end date in `YYYY-MM-DD` format.
    pub expiration_date: Option<NaiveDate>,
}

/// Type of insurance coverage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
#[baml(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum CoverageType {
    Property,
    Auto,
    Liability,
    Health,
    Travel,
    Other,
}

/// An insured object involved in the claim (vehicle, building, person, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
pub struct InsuredObject {
    /// Unique identifier for insured object.
    ///
    /// For vehicles, use VIN format (e.g., `VIN12345678901234567`).
    /// For buildings, use `PROP-XXXXXX` format.
    /// For liability, use `LIAB-XXXXXX` format.
    /// For other objects, use `OBJ-XXXXXX` format,
    /// where `X` is a digit.
    pub object_id: Option<String>,

    /// Type of insured object.
    pub object_type: InsuredObjectType,

    /// Make and model for vehicles (use standardized manufacturer names and models),
    /// or building type for property.
    pub make_model: Option<String>,

    /// Year for vehicles or year built for buildings.
    pub year: Option<i32>,

    /// Full street address where object is located or originated from.
    pub location_address: Option<String>,

    /// Estimated monetary value in USD without currency symbol.
    pub estimated_value: Option<i64>,
}

/// Type of insured object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
#[baml(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum InsuredObjectType {
    Vehicle,
    Building,
    Person,
    Other,
}

/// Structured incident details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
pub struct IncidentDescription {
    /// Specific standardized incident type.
    pub incident_type: IncidentType,

    /// Standardized location type where incident occurred.
    pub location_type: LocationType,

    /// Estimated damage in USD without currency symbol.
    pub estimated_damage_amount: Option<i64>,

    /// Police report number if applicable.
    pub police_report_number: Option<String>,
}

/// Specific standardized incident type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
#[baml(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum IncidentType {
    RearEndCollision,
    SideImpactCollision,
    HeadOnCollision,
    ParkingLotCollision,
    HouseFire,
    KitchenFire,
    ElectricalFire,
    BurstPipeFlood,
    StormDamage,
    RoofLeak,
    SlipAndFall,
    PropertyInjury,
    ProductLiability,
    TheftBurglary,
    Vandalism,
}

/// Standardized location type where incident occurred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
#[baml(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum LocationType {
    Intersection,
    Highway,
    ParkingLot,
    Driveway,
    ResidentialStreet,
    ResidenceInterior,
    ResidenceExterior,
    CommercialProperty,
    PublicProperty,
}

/// Top-level insurance claim object aggregating all extracted fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, BamlType)]
pub struct InsuranceClaim {
    /// Basic claim information.
    pub header: ClaimHeader,

    /// Policy information if available.
    pub policy_details: Option<PolicyDetails>,

    /// List of insured objects involved, if applicable.
    pub insured_objects: Option<Vec<InsuredObject>>,

    /// Structured incident details.
    pub incident_description: Option<IncidentDescription>,
}

#[derive(Signature, Clone, Debug)]
/// Extract the insurance claim information from the following text.
/// - If you are unsure about a field, leave it as null.
pub struct InsuranceClaimInfo {
    #[input]
    claim_text: String,

    #[output]
    claim: InsuranceClaim,
}

struct Args {
    start: usize,
    end: Option<usize>,
    input: String,
    output: String,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    reasoning_effort: Option<String>,
}

fn print_help() {
    println!(
        "Usage: 17-insurance-claims-extract [options]\n\n\
Options:\n\
  -s, --start <n>         Start index (1-based, default: 1)\n\
  -e, --end <n>           End index (inclusive, default: dataset length)\n\
  -i, --input <path>      Input parquet file path\n\
  -o, --output <path>     Output NDJSON path\n\
  -m, --model <name>      Model name (default: openai-responses:gpt-5.2)\n\
  -t, --temperature <f>   Sampling temperature (default: LM default)\n\
  --max-tokens <n>        Max output tokens (default: LM default)\n\
  --reasoning-effort <v>  OpenAI Responses reasoning effort (minimal|low|medium|high)\n\
  -h, --help              Show this help message\n"
    );
}

fn is_openai_responses_model(model: &str) -> bool {
    model.starts_with("openai-responses:")
        || model.starts_with("openai_responses:")
        || model.starts_with("openai.responses:")
}

fn parse_error_to_json(error: &dspy_rs::ParseError) -> Value {
    match error {
        dspy_rs::ParseError::MissingField { field, .. } => {
            json!({"type": "missing_field", "field": field})
        }
        dspy_rs::ParseError::ExtractionFailed { field, reason, .. } => {
            json!({"type": "extraction_failed", "field": field, "reason": reason})
        }
        dspy_rs::ParseError::CoercionFailed {
            field,
            expected_type,
            raw_text,
            source,
        } => json!({
            "type": "coercion_failed",
            "field": field,
            "expected_type": expected_type,
            "raw_text": raw_text,
            "reason": source.to_string(),
        }),
        dspy_rs::ParseError::AssertFailed {
            field,
            label,
            expression,
            value,
        } => json!({
            "type": "assert_failed",
            "field": field,
            "label": label,
            "expression": expression,
            "value": value,
        }),
        dspy_rs::ParseError::Multiple { errors, partial } => {
            let mut value = json!({
                "type": "multiple",
                "errors": errors.iter().map(parse_error_to_json).collect::<Vec<_>>(),
            });
            if let Some(partial) = partial {
                if let Value::Object(map) = &mut value {
                    if let Ok(partial_value) = serde_json::to_value(partial) {
                        map.insert("partial".to_string(), partial_value);
                    }
                }
            }
            value
        }
    }
}

fn parse_args() -> Result<Args> {
    let mut start = 1usize;
    let mut end: Option<usize> = None;
    let mut input = "../data/insurance_claims_extraction.parquet".to_string();
    let mut output = "../data/structured_output_dsrs.json".to_string();
    let mut model = "openai-responses:gpt-5.2".to_string();
    let mut temperature = None;
    let mut max_tokens = None;
    let mut reasoning_effort = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-s" | "--start" => {
                start = args
                    .next()
                    .context("missing value for --start")?
                    .parse()
                    .context("invalid value for --start")?;
            }
            "-e" | "--end" => {
                end = Some(
                    args.next()
                        .context("missing value for --end")?
                        .parse()
                        .context("invalid value for --end")?,
                );
            }
            "-i" | "--input" => {
                input = args.next().context("missing value for --input")?;
            }
            "-o" | "--output" => {
                output = args.next().context("missing value for --output")?;
            }
            "-m" | "--model" => {
                model = args.next().context("missing value for --model")?;
            }
            "-t" | "--temperature" => {
                temperature = Some(
                    args.next()
                        .context("missing value for --temperature")?
                        .parse()
                        .context("invalid value for --temperature")?,
                );
            }
            "--max-tokens" => {
                max_tokens = Some(
                    args.next()
                        .context("missing value for --max-tokens")?
                        .parse()
                        .context("invalid value for --max-tokens")?,
                );
            }
            "--reasoning-effort" => {
                reasoning_effort = Some(args.next().context("missing value for --reasoning-effort")?);
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            unknown => bail!("Unknown argument: {unknown}"),
        }
    }

    Ok(Args {
        start,
        end,
        input,
        output,
        model,
        temperature,
        max_tokens,
        reasoning_effort,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let Args {
        start,
        end,
        input,
        output,
        model,
        temperature,
        max_tokens,
        reasoning_effort,
    } = parse_args()?;

    let additional_params = if let Some(effort) = reasoning_effort.as_deref() {
        let effort_value = match effort {
            "minimal" | "low" | "medium" | "high" => effort,
            _ => bail!(
                "Invalid value for --reasoning-effort: {effort}. Use minimal|low|medium|high."
            ),
        };
        if !is_openai_responses_model(&model) {
            bail!("--reasoning-effort requires --model openai-responses:<model>");
        }
        Some(json!({
            "reasoning": {
                "effort": effort_value,
            }
        }))
    } else {
        None
    };

    let lm = match (temperature, max_tokens, additional_params) {
        (Some(temperature), Some(max_tokens), Some(params)) => {
            LM::builder()
                .model(model.clone())
                .temperature(temperature)
                .max_tokens(max_tokens)
                .additional_params(params)
                .build()
                .await?
        }
        (Some(temperature), Some(max_tokens), None) => {
            LM::builder()
                .model(model.clone())
                .temperature(temperature)
                .max_tokens(max_tokens)
                .build()
                .await?
        }
        (Some(temperature), None, Some(params)) => {
            LM::builder()
                .model(model.clone())
                .temperature(temperature)
                .additional_params(params)
                .build()
                .await?
        }
        (None, Some(max_tokens), Some(params)) => {
            LM::builder()
                .model(model.clone())
                .max_tokens(max_tokens)
                .additional_params(params)
                .build()
                .await?
        }
        (Some(temperature), None, None) => {
            LM::builder()
                .model(model.clone())
                .temperature(temperature)
                .build()
                .await?
        }
        (None, Some(max_tokens), None) => {
            LM::builder()
                .model(model.clone())
                .max_tokens(max_tokens)
                .build()
                .await?
        }
        (None, None, Some(params)) => {
            LM::builder()
                .model(model.clone())
                .additional_params(params)
                .build()
                .await?
        }
        (None, None, None) => LM::builder().model(model.clone()).build().await?,
    };

    configure(lm, ChatAdapter);

    let examples = DataLoader::load_parquet(&input, vec!["claim_text".to_string()], vec![])?;
    let total = examples.len();
    let end = end.unwrap_or(total).min(total);

    if start < 1 || start > end {
        bail!("Start index must be >= 1 and <= end index.");
    }

    if let Some(parent) = Path::new(&output).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = fs::File::create(&output)?;
    let mut writer = BufWriter::new(file);
    let predictor = Predict::<InsuranceClaimInfo>::builder().build();

    for (idx, example) in examples.iter().enumerate() {
        let record_id = idx + 1;
        if record_id < start || record_id > end {
            continue;
        }

        let claim_text = example
            .get("claim_text", None)
            .as_str()
            .unwrap_or_default()
            .to_string();

        if claim_text.is_empty() {
            let value = json!({
                "record_id": record_id,
                "error": "missing claim_text",
            });
            writeln!(writer, "{}", serde_json::to_string(&value)?)?;
            eprintln!("Record {record_id} skipped: missing claim_text");
            continue;
        }

        match predictor.call(InsuranceClaimInfoInput { claim_text }).await {
            Ok(result) => {
                let mut value = serde_json::to_value(&result.output.claim)
                    .context("serialize claim output")?;
                if let serde_json::Value::Object(ref mut map) = value {
                    map.insert("record_id".to_string(), json!(record_id));
                    map.insert("raw_response".to_string(), json!(result.raw_response));
                }
                writeln!(writer, "{}", serde_json::to_string(&value)?)?;
                println!("Record {record_id} completed");
            }
            Err(err) => {
                let error_detail = match &err {
                    PredictError::Lm { source } => source.to_string(),
                    PredictError::Parse { source, .. } => source.to_string(),
                    PredictError::Conversion { source, .. } => source.to_string(),
                };
                let mut value = json!({
                    "record_id": record_id,
                    "error": err.to_string(),
                    "raw_response": Value::Null,
                });
                if let Value::Object(ref mut map) = value {
                    map.insert("error_detail".to_string(), json!(error_detail));
                    match &err {
                        PredictError::Parse {
                            source,
                            raw_response,
                            ..
                        } => {
                            map.insert("raw_response".to_string(), json!(raw_response));
                            map.insert("parse_error".to_string(), parse_error_to_json(source));
                        }
                        PredictError::Conversion { parsed, .. } => {
                            if let Ok(parsed_value) = serde_json::to_value(parsed) {
                                map.insert("parsed".to_string(), parsed_value);
                            }
                        }
                        PredictError::Lm { .. } => {}
                    }
                }
                writeln!(writer, "{}", serde_json::to_string(&value)?)?;
                eprintln!("Record {record_id} failed: {error_detail}");
            }
        }
    }

    writer.flush()?;
    println!("Saved results to {}", output);
    Ok(())
}
