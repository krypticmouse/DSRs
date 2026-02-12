use anyhow::{Context, Result, anyhow};
use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use bamltype::baml_types::BamlMap;
use csv::{ReaderBuilder, StringRecord};
use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use reqwest;
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tracing::debug;

use crate::data::utils::is_url;
use crate::predictors::Example as TypedExample;
use crate::{BamlType, BamlValue, Signature};

/// Controls how typed loaders handle source fields that are not part of the target signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownFieldPolicy {
    /// Ignore extra source fields that are not consumed by the signature.
    #[default]
    Ignore,
    /// Fail the load when a row contains any extra source field.
    Error,
}

/// Options for schema-driven typed loading.
///
/// `field_map` remaps signature fields to source fields:
/// - key: signature field name (`S::schema()` field rust name)
/// - value: source field/column name in the file/dataset
///
/// `unknown_fields` controls whether extra source fields are ignored or rejected.
#[derive(Debug, Clone)]
pub struct TypedLoadOptions {
    pub field_map: HashMap<String, String>,
    pub unknown_fields: UnknownFieldPolicy,
}

impl Default for TypedLoadOptions {
    fn default() -> Self {
        Self {
            field_map: HashMap::new(),
            unknown_fields: UnknownFieldPolicy::Ignore,
        }
    }
}

/// Raw parsed row passed to custom mapper closures in `load_*_with` APIs.
///
/// Values are normalized into `serde_json::Value` so mappers can deserialize
/// directly into strongly typed Rust values using [`RowRecord::get`].
#[derive(Debug, Clone)]
pub struct RowRecord {
    /// 1-based row index in the loaded stream after filtering empty rows.
    pub row_index: usize,
    /// Parsed key-value payload for the row.
    pub values: HashMap<String, serde_json::Value>,
}

impl RowRecord {
    /// Deserialize a typed value from a row field.
    ///
    /// Returns [`DataLoadError::MissingField`] if the key is absent.
    ///
    /// Returns [`DataLoadError::TypeMismatch`] on deserialization failure.
    /// For ergonomic CSV mapping, `String` reads will coerce scalar JSON values
    /// (number/bool) into strings.
    pub fn get<T: serde::de::DeserializeOwned + 'static>(
        &self,
        key: &str,
    ) -> std::result::Result<T, DataLoadError> {
        let value = self
            .values
            .get(key)
            .ok_or_else(|| DataLoadError::MissingField {
                row: self.row_index,
                field: key.to_string(),
            })?;

        match serde_json::from_value::<T>(value.clone()) {
            Ok(parsed) => Ok(parsed),
            Err(err) => {
                if TypeId::of::<T>() == TypeId::of::<String>() {
                    let coerced = match value {
                        serde_json::Value::String(text) => text.clone(),
                        serde_json::Value::Number(number) => number.to_string(),
                        serde_json::Value::Bool(flag) => flag.to_string(),
                        other => other.to_string(),
                    };
                    return serde_json::from_value::<T>(serde_json::Value::String(coerced))
                        .map_err(|fallback_err| DataLoadError::TypeMismatch {
                            row: self.row_index,
                            field: key.to_string(),
                            message: fallback_err.to_string(),
                        });
                }

                Err(DataLoadError::TypeMismatch {
                    row: self.row_index,
                    field: key.to_string(),
                    message: err.to_string(),
                })
            }
        }
    }
}

/// Row-aware errors produced by typed data loading.
#[derive(Debug, thiserror::Error)]
pub enum DataLoadError {
    /// Source read/download failure.
    #[error("I/O error: {0}")]
    Io(anyhow::Error),
    /// CSV parser failure.
    #[error("CSV error: {0}")]
    Csv(anyhow::Error),
    /// JSON/JSONL parser failure.
    #[error("JSON error: {0}")]
    Json(anyhow::Error),
    /// Parquet parser failure.
    #[error("Parquet error: {0}")]
    Parquet(anyhow::Error),
    /// HuggingFace Hub listing or file retrieval failure.
    #[error("HuggingFace error: {0}")]
    Hf(anyhow::Error),
    /// Required signature field was missing from a row.
    #[error("missing field `{field}` at row {row}")]
    MissingField { row: usize, field: String },
    /// Row had an unexpected extra field when unknown-field policy is `Error`.
    #[error("unknown field `{field}` at row {row}")]
    UnknownField { row: usize, field: String },
    /// Field existed but could not be converted to required type.
    #[error("type mismatch for field `{field}` at row {row}: {message}")]
    TypeMismatch {
        row: usize,
        field: String,
        message: String,
    },
    /// Custom mapper closure returned an error.
    #[error("mapper error at row {row}: {message}")]
    Mapper { row: usize, message: String },
}

/// Typed dataset ingress for JSON/CSV/Parquet/HuggingFace sources.
///
/// Canonical public contract:
/// - Returns `Vec<Example<S>>` directly.
/// - Uses `S::schema()` for required input/output fields.
/// - Supports field remapping via [`TypedLoadOptions::field_map`].
/// - Reports row-aware failures through [`DataLoadError`].
pub struct DataLoader;

impl DataLoader {
    #[tracing::instrument(
        name = "dsrs.data.load_json",
        level = "debug",
        skip(opts),
        fields(
            is_url = is_url(path),
            lines,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from JSON array/object or JSONL.
    ///
    /// `lines = true` treats the file as JSONL (`one object per line`).
    ///
    /// # Errors
    /// Returns [`DataLoadError`] wrapped in `anyhow::Error` for parse, schema,
    /// mapping, and conversion failures.
    pub fn load_json<S: Signature>(
        path: &str,
        lines: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let rows = Self::load_json_rows(path, lines)?;
        let examples = Self::rows_to_typed::<S>(rows, &opts)?;
        debug!(examples = examples.len(), "typed json examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_json_with",
        level = "debug",
        skip(opts, mapper),
        fields(
            is_url = is_url(path),
            lines,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from JSON/JSONL and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_json_with<S, F>(
        path: &str,
        lines: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S: Signature,
        F: Fn(&RowRecord) -> Result<TypedExample<S>>,
    {
        let _ = opts;
        let rows = Self::load_json_rows(path, lines)?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed json examples loaded via mapper"
        );
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_csv",
        level = "debug",
        skip(opts),
        fields(
            is_url = is_url(path),
            delimiter,
            has_headers,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from CSV.
    ///
    /// When `has_headers` is `false`, fields are exposed as `column_{idx}` for
    /// mapper-based paths. Signature-based paths should typically use headers.
    pub fn load_csv<S: Signature>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let rows = Self::load_csv_rows(path, delimiter, has_headers)?;
        let examples = Self::rows_to_typed::<S>(rows, &opts)?;
        debug!(examples = examples.len(), "typed csv examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_csv_with",
        level = "debug",
        skip(opts, mapper),
        fields(
            is_url = is_url(path),
            delimiter,
            has_headers,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from CSV and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_csv_with<S, F>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S: Signature,
        F: Fn(&RowRecord) -> Result<TypedExample<S>>,
    {
        let _ = opts;
        let rows = Self::load_csv_rows(path, delimiter, has_headers)?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed csv examples loaded via mapper"
        );
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_parquet",
        level = "debug",
        skip(opts),
        fields(
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from a local Parquet file.
    pub fn load_parquet<S: Signature>(
        path: &str,
        opts: TypedLoadOptions,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let rows = Self::load_parquet_rows(Path::new(path))?;
        let examples = Self::rows_to_typed::<S>(rows, &opts)?;
        debug!(examples = examples.len(), "typed parquet examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_parquet_with",
        level = "debug",
        skip(opts, mapper),
        fields(
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from Parquet and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_parquet_with<S, F>(
        path: &str,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S: Signature,
        F: Fn(&RowRecord) -> Result<TypedExample<S>>,
    {
        let _ = opts;
        let rows = Self::load_parquet_rows(Path::new(path))?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed parquet examples loaded via mapper"
        );
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_hf",
        level = "debug",
        skip(opts),
        fields(
            dataset = dataset_name,
            subset,
            split,
            verbose,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from a HuggingFace dataset split.
    ///
    /// Supports Parquet, JSON/JSONL, and CSV artifacts discovered in the dataset
    /// repo. `subset` and `split` are substring filters on artifact filenames.
    pub fn load_hf<S: Signature>(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let rows = Self::load_hf_rows(dataset_name, subset, split, verbose)?;
        let examples = Self::rows_to_typed::<S>(rows, &opts)?;
        debug!(examples = examples.len(), "typed hf examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_hf_with",
        level = "debug",
        skip(opts, mapper),
        fields(
            dataset = dataset_name,
            subset,
            split,
            verbose,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from HuggingFace and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_hf_with<S, F>(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S: Signature,
        F: Fn(&RowRecord) -> Result<TypedExample<S>>,
    {
        let _ = opts;
        let rows = Self::load_hf_rows(dataset_name, subset, split, verbose)?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed hf examples loaded via mapper"
        );
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_hf_from_parquet",
        level = "debug",
        skip(parquet_files, opts),
        fields(
            files = parquet_files.len(),
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from a local set of Parquet files.
    ///
    /// This is primarily used for deterministic/offline testing of HF-like data
    /// ingestion flows without network calls.
    pub fn load_hf_from_parquet<S: Signature>(
        parquet_files: Vec<PathBuf>,
        opts: TypedLoadOptions,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let rows = Self::load_rows_from_parquet_files(&parquet_files)?;
        let examples = Self::rows_to_typed::<S>(rows, &opts)?;
        debug!(
            examples = examples.len(),
            "typed hf parquet examples loaded"
        );
        Ok(examples)
    }

    fn rows_to_typed<S: Signature>(
        rows: Vec<RowRecord>,
        opts: &TypedLoadOptions,
    ) -> Result<Vec<TypedExample<S>>>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        rows.into_iter()
            .map(|row| typed_example_from_row::<S>(&row, opts).map_err(anyhow::Error::from))
            .collect()
    }

    fn rows_with_mapper<S, F>(rows: Vec<RowRecord>, mapper: F) -> Result<Vec<TypedExample<S>>>
    where
        S: Signature,
        F: Fn(&RowRecord) -> Result<TypedExample<S>>,
    {
        rows.into_iter()
            .map(|row| {
                mapper(&row).map_err(|err| DataLoadError::Mapper {
                    row: row.row_index,
                    message: err.to_string(),
                })
            })
            .map(|result| result.map_err(anyhow::Error::from))
            .collect()
    }

    fn fetch_text(path: &str) -> std::result::Result<String, DataLoadError> {
        if is_url(path) {
            let response = reqwest::blocking::get(path)
                .with_context(|| format!("failed to GET `{path}`"))
                .map_err(DataLoadError::Io)?;
            response.text().map_err(|err| DataLoadError::Io(err.into()))
        } else {
            fs::read_to_string(path).map_err(|err| DataLoadError::Io(err.into()))
        }
    }

    fn load_json_rows(
        path: &str,
        lines: bool,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let data = Self::fetch_text(path)?;

        if lines {
            let mut rows = Vec::new();
            for (idx, line) in data.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let value: serde_json::Value =
                    serde_json::from_str(line).map_err(|err| DataLoadError::Json(anyhow!(err)))?;
                rows.push(row_from_json_value(value, idx + 1)?);
            }
            debug!(rows = rows.len(), "jsonl rows loaded");
            return Ok(rows);
        }

        let value: serde_json::Value =
            serde_json::from_str(&data).map_err(|err| DataLoadError::Json(anyhow!(err)))?;

        let rows = match value {
            serde_json::Value::Array(items) => items
                .into_iter()
                .enumerate()
                .map(|(idx, item)| row_from_json_value(item, idx + 1))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            other => vec![row_from_json_value(other, 1)?],
        };

        debug!(rows = rows.len(), "json rows loaded");
        Ok(rows)
    }

    fn load_csv_rows(
        path: &str,
        delimiter: char,
        has_headers: bool,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        if is_url(path) {
            let bytes = reqwest::blocking::get(path)
                .with_context(|| format!("failed to GET `{path}`"))
                .map_err(DataLoadError::Csv)?
                .bytes()
                .map_err(|err| DataLoadError::Csv(err.into()))?
                .to_vec();

            let cursor = Cursor::new(bytes);
            let mut reader = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_reader(cursor);
            return Self::collect_csv_rows(&mut reader, has_headers);
        }

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter as u8)
            .has_headers(has_headers)
            .from_path(path)
            .map_err(|err| DataLoadError::Csv(err.into()))?;
        Self::collect_csv_rows(&mut reader, has_headers)
    }

    fn collect_csv_rows<R: std::io::Read>(
        reader: &mut csv::Reader<R>,
        has_headers: bool,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let header_names = if has_headers {
            Some(
                reader
                    .headers()
                    .map_err(|err| DataLoadError::Csv(err.into()))?
                    .iter()
                    .map(|header| header.to_string())
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let rows = reader
            .records()
            .enumerate()
            .map(|(idx, record)| {
                let record = record.map_err(|err| DataLoadError::Csv(err.into()))?;
                Ok(csv_record_to_row_record(
                    &record,
                    idx + 1,
                    header_names.as_deref(),
                ))
            })
            .collect::<std::result::Result<Vec<_>, DataLoadError>>()?;

        debug!(rows = rows.len(), "csv rows loaded");
        Ok(rows)
    }

    fn load_parquet_rows(path: &Path) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let file = fs::File::open(path).map_err(|err| DataLoadError::Parquet(err.into()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|err| DataLoadError::Parquet(err.into()))?;
        let reader = builder
            .build()
            .map_err(|err| DataLoadError::Parquet(err.into()))?;

        let mut rows = Vec::new();
        let mut row_index = 1usize;

        for batch_result in reader {
            let batch = batch_result.map_err(|err| DataLoadError::Parquet(err.into()))?;
            let schema = batch.schema();

            for local_row in 0..batch.num_rows() {
                let mut values = HashMap::new();

                for col_idx in 0..batch.num_columns() {
                    let column = batch.column(col_idx);
                    let field_name = schema.field(col_idx).name().to_string();

                    if let Some(value) = parquet_value_to_json(column.as_ref(), local_row) {
                        values.insert(field_name, value);
                    }
                }

                if !values.is_empty() {
                    rows.push(RowRecord { row_index, values });
                }
                row_index += 1;
            }
        }

        debug!(rows = rows.len(), "parquet rows loaded");
        Ok(rows)
    }

    fn load_rows_from_parquet_files(
        parquet_files: &[PathBuf],
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let mut all_rows = Vec::new();
        let mut next_index = 1usize;

        for file in parquet_files {
            let mut rows = Self::load_parquet_rows(file)?;
            for row in &mut rows {
                row.row_index = next_index;
                next_index += 1;
            }
            all_rows.extend(rows);
        }

        Ok(all_rows)
    }

    fn load_hf_rows(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let api = Api::new().map_err(|err| DataLoadError::Hf(err.into()))?;
        let repo = api.dataset(dataset_name.to_string());
        let metadata = repo.info().map_err(|err| DataLoadError::Hf(err.into()))?;

        let mut rows = Vec::new();
        let mut next_index = 1usize;

        for sibling in metadata.siblings {
            let file = sibling.rfilename;

            if (!subset.is_empty() && !file.contains(subset))
                || (!split.is_empty() && !file.contains(split))
            {
                continue;
            }

            let supported = file.ends_with(".parquet")
                || file.ends_with(".json")
                || file.ends_with(".jsonl")
                || file.ends_with(".csv");
            if !supported {
                continue;
            }

            let file_path = repo
                .get(&file)
                .map_err(|err| DataLoadError::Hf(err.into()))?;
            let path_str = file_path
                .to_str()
                .ok_or_else(|| DataLoadError::Io(anyhow!("invalid UTF-8 file path")))?;

            if verbose {
                println!("Loading file: {path_str}");
            }

            let mut file_rows = if file.ends_with(".parquet") {
                Self::load_parquet_rows(&file_path)?
            } else if file.ends_with(".json") || file.ends_with(".jsonl") {
                Self::load_json_rows(path_str, file.ends_with(".jsonl"))?
            } else {
                Self::load_csv_rows(path_str, ',', true)?
            };

            for row in &mut file_rows {
                row.row_index = next_index;
                next_index += 1;
            }

            rows.extend(file_rows);
        }

        if verbose {
            println!("Loaded {} rows", rows.len());
        }

        debug!(rows = rows.len(), "hf rows loaded");
        Ok(rows)
    }
}

fn resolve_source_field<'a>(field: &'a str, opts: &'a TypedLoadOptions) -> &'a str {
    opts.field_map
        .get(field)
        .map(String::as_str)
        .unwrap_or(field)
}

fn typed_example_from_row<S: Signature>(
    row: &RowRecord,
    opts: &TypedLoadOptions,
) -> std::result::Result<TypedExample<S>, DataLoadError>
where
    S::Input: BamlType,
    S::Output: BamlType,
{
    let schema = S::schema();
    let mut used_source_fields = HashSet::new();

    let input_map = baml_map_for_fields(
        row,
        schema
            .input_fields()
            .iter()
            .map(|field| field.rust_name.as_str()),
        opts,
        &mut used_source_fields,
    )?;

    let output_map = baml_map_for_fields(
        row,
        schema
            .output_fields()
            .iter()
            .map(|field| field.rust_name.as_str()),
        opts,
        &mut used_source_fields,
    )?;

    if opts.unknown_fields == UnknownFieldPolicy::Error {
        for key in row.values.keys() {
            if !used_source_fields.contains(key) {
                return Err(DataLoadError::UnknownField {
                    row: row.row_index,
                    field: key.clone(),
                });
            }
        }
    }

    let input = S::Input::try_from_baml_value(BamlValue::Map(input_map)).map_err(|err| {
        DataLoadError::TypeMismatch {
            row: row.row_index,
            field: "input".to_string(),
            message: err.to_string(),
        }
    })?;

    let output = S::Output::try_from_baml_value(BamlValue::Map(output_map)).map_err(|err| {
        DataLoadError::TypeMismatch {
            row: row.row_index,
            field: "output".to_string(),
            message: err.to_string(),
        }
    })?;

    Ok(TypedExample::new(input, output))
}

fn baml_map_for_fields<'a>(
    row: &RowRecord,
    signature_fields: impl Iterator<Item = &'a str>,
    opts: &TypedLoadOptions,
    used_source_fields: &mut HashSet<String>,
) -> std::result::Result<BamlMap<String, BamlValue>, DataLoadError> {
    let mut map = BamlMap::new();

    for signature_field in signature_fields {
        let source_field = resolve_source_field(signature_field, opts);
        let value = row
            .values
            .get(source_field)
            .ok_or_else(|| DataLoadError::MissingField {
                row: row.row_index,
                field: signature_field.to_string(),
            })?;

        let baml_value =
            BamlValue::try_from(value.clone()).map_err(|err| DataLoadError::TypeMismatch {
                row: row.row_index,
                field: signature_field.to_string(),
                message: err.to_string(),
            })?;

        map.insert(signature_field.to_string(), baml_value);
        used_source_fields.insert(source_field.to_string());
    }

    Ok(map)
}

fn row_from_json_value(
    value: serde_json::Value,
    row_index: usize,
) -> std::result::Result<RowRecord, DataLoadError> {
    let object = value.as_object().ok_or_else(|| {
        DataLoadError::Json(anyhow!(
            "row {row_index}: expected JSON object, got {}",
            value
        ))
    })?;

    Ok(RowRecord {
        row_index,
        values: object.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    })
}

fn parse_csv_cell(cell: &str) -> serde_json::Value {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return serde_json::Value::String(String::new());
    }

    serde_json::from_str::<serde_json::Value>(trimmed)
        .unwrap_or_else(|_| serde_json::Value::String(cell.to_string()))
}

fn csv_record_to_row_record(
    record: &StringRecord,
    row_index: usize,
    headers: Option<&[String]>,
) -> RowRecord {
    let mut values = HashMap::new();

    for (idx, cell) in record.iter().enumerate() {
        let key = headers
            .and_then(|items| items.get(idx))
            .cloned()
            .unwrap_or_else(|| format!("column_{idx}"));
        values.insert(key, parse_csv_cell(cell));
    }

    RowRecord { row_index, values }
}

fn parquet_value_to_json(column: &dyn Array, row_idx: usize) -> Option<serde_json::Value> {
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<BooleanArray>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Int16Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Int8Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt64Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt32Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt16Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt8Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
        return (!values.is_null(row_idx)).then(|| serde_json::json!(values.value(row_idx)));
    }

    None
}
