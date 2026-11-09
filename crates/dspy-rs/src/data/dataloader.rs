use anyhow::{Context, Result, anyhow};
#[cfg(feature = "data")]
use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
#[cfg(feature = "data")]
use csv::{ReaderBuilder, StringRecord};
#[cfg(feature = "data")]
use hf_hub::api::sync::Api;
#[cfg(feature = "data")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use reqwest;
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::cell::Cell;
use std::fs;
#[cfg(feature = "data")]
use std::io::Cursor;
use std::io::Read;
#[cfg(feature = "data")]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, SeqAccess, Visitor};
use tracing::debug;

use crate::data::utils::is_url;
use crate::typesys::schema::field_type_from_shape;
use crate::FieldType;
use facet::{Facet, Type, UserType};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

/// Controls how typed loaders handle source fields that are not part of the target row struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownFieldPolicy {
    /// Ignore extra source fields that are not consumed by the signature.
    #[default]
    Ignore,
    /// Fail the load when a row contains any extra source field.
    Error,
}

/// Options for shape-driven typed loading.
///
/// `field_map` remaps row-struct fields to source fields:
/// - key: row struct field name
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

/// Limits applied to HTTP JSON and CSV dataset loads. Local files are unaffected.
#[derive(Debug, Clone, Copy)]
pub struct RemoteLoadLimits {
    pub max_bytes: usize,
    pub max_rows: usize,
    pub timeout: Duration,
}

impl Default for RemoteLoadLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_rows: 50_000,
            timeout: Duration::from_secs(30),
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
    /// Required row-struct field was missing from a row.
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
/// JSON/JSONL loading is always available; the CSV, Parquet, and HuggingFace
/// loaders require the (default-on) `data` feature, which carries the heavy
/// arrow/parquet/hf-hub/csv dependency stack.
///
/// Canonical public contract:
/// - Returns `Vec<E>` for any row struct `E: Deserialize + Facet` — rows are
///   *your* type, not a signature-shaped pair, so they can carry gold labels
///   and metric-only fields freely.
/// - Uses `E`'s Facet shape for required fields and type-aware coercion.
/// - Supports field remapping via [`TypedLoadOptions::field_map`].
/// - Reports row-aware failures through [`DataLoadError`].
pub struct DataLoader;

impl DataLoader {
    #[tracing::instrument(
        name = "dsrs.data.load_json",
        level = "debug",
        skip(path, opts),
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
    pub fn load_json<E>(path: &str, lines: bool, opts: TypedLoadOptions) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        Self::load_json_with_limits(path, lines, opts, RemoteLoadLimits::default())
    }

    /// Load JSON with custom limits for HTTP sources.
    pub fn load_json_with_limits<E>(
        path: &str,
        lines: bool,
        opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let rows = Self::load_json_rows(path, lines, limits)?;
        let examples = Self::rows_to_typed::<E>(rows, &opts)?;
        debug!(examples = examples.len(), "typed json examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_json_with",
        level = "debug",
        skip(path, opts, mapper),
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
    pub fn load_json_with<E, F>(
        path: &str,
        lines: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        Self::load_json_with_mapper_limits(path, lines, opts, RemoteLoadLimits::default(), mapper)
    }

    /// Map JSON rows with custom limits for HTTP sources.
    pub fn load_json_with_mapper_limits<E, F>(
        path: &str,
        lines: bool,
        _opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        let rows = Self::load_json_rows(path, lines, limits)?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed json examples loaded via mapper"
        );
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_csv",
        level = "debug",
        skip(path, opts),
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
    /// mapper-based paths. Shape-based paths should typically use headers.
    pub fn load_csv<E>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        Self::load_csv_with_limits(
            path,
            delimiter,
            has_headers,
            opts,
            RemoteLoadLimits::default(),
        )
    }

    /// Load CSV with custom limits for HTTP sources.
    #[cfg(feature = "data")]
    pub fn load_csv_with_limits<E>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let rows = Self::load_csv_rows(path, delimiter, has_headers, limits)?;
        let examples = Self::rows_to_typed::<E>(rows, &opts)?;
        debug!(examples = examples.len(), "typed csv examples loaded");
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_csv_with",
        level = "debug",
        skip(path, opts, mapper),
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
    pub fn load_csv_with<E, F>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        Self::load_csv_with_mapper_limits(
            path,
            delimiter,
            has_headers,
            opts,
            RemoteLoadLimits::default(),
            mapper,
        )
    }

    /// Map CSV rows with custom limits for HTTP sources.
    #[cfg(feature = "data")]
    pub fn load_csv_with_mapper_limits<E, F>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        _opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        let rows = Self::load_csv_rows(path, delimiter, has_headers, limits)?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed csv examples loaded via mapper"
        );
        Ok(examples)
    }

    #[cfg(feature = "data")]
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
    pub fn load_parquet<E>(
        path: &str,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let rows = Self::load_parquet_rows(Path::new(path))?;
        let examples = Self::rows_to_typed::<E>(rows, &opts)?;
        debug!(examples = examples.len(), "typed parquet examples loaded");
        Ok(examples)
    }

    #[cfg(feature = "data")]
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
    pub fn load_parquet_with<E, F>(
        path: &str,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
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

    #[cfg(feature = "data")]
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
    pub fn load_hf<E>(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let rows = Self::load_hf_rows(dataset_name, subset, split, verbose)?;
        let examples = Self::rows_to_typed::<E>(rows, &opts)?;
        debug!(examples = examples.len(), "typed hf examples loaded");
        Ok(examples)
    }

    #[cfg(feature = "data")]
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
    pub fn load_hf_with<E, F>(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
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

    #[cfg(feature = "data")]
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
    pub fn load_hf_from_parquet<E>(
        parquet_files: Vec<PathBuf>,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let rows = Self::load_rows_from_parquet_files(&parquet_files)?;
        let examples = Self::rows_to_typed::<E>(rows, &opts)?;
        debug!(
            examples = examples.len(),
            "typed hf parquet examples loaded"
        );
        Ok(examples)
    }

    fn rows_to_typed<E>(rows: Vec<RowRecord>, opts: &TypedLoadOptions) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let fields = row_struct_fields::<E>()?;
        rows.into_iter()
            .map(|row| struct_from_row::<E>(&row, &fields, opts).map_err(anyhow::Error::from))
            .collect()
    }

    fn rows_with_mapper<E, F>(rows: Vec<RowRecord>, mapper: F) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
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

    fn fetch_remote_bytes(
        path: &str,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<(Vec<u8>, Option<String>), DataLoadError> {
        let started = Instant::now();
        let client = reqwest::blocking::Client::builder()
            .timeout(limits.timeout)
            .build()
            .map_err(|err| DataLoadError::Io(err.into()))?;
        let mut response = client
            .get(path)
            .send()
            .context("failed to fetch remote dataset")
            .map_err(DataLoadError::Io)?;
        if started.elapsed() >= limits.timeout {
            return Err(remote_timeout(limits.timeout));
        }
        let charset = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| {
                value.split(';').skip(1).find_map(|part| {
                    let (key, value) = part.trim().split_once('=')?;
                    key.trim().eq_ignore_ascii_case("charset").then(|| value.trim().trim_matches('"').to_owned())
                })
            });
        let max_bytes = u64::try_from(limits.max_bytes).unwrap_or(u64::MAX);
        if response.content_length().is_some_and(|len| len > max_bytes) {
            return Err(remote_limit("bytes", limits.max_bytes));
        }
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            if started.elapsed() >= limits.timeout {
                return Err(remote_timeout(limits.timeout));
            }
            let remaining = limits.max_bytes.saturating_sub(bytes.len());
            let read_len = remaining.saturating_add(1).min(chunk.len());
            let count = response
                .read(&mut chunk[..read_len])
                .map_err(|err| DataLoadError::Io(err.into()))?;
            if started.elapsed() >= limits.timeout {
                return Err(remote_timeout(limits.timeout));
            }
            if count == 0 {
                break;
            }
            if count > remaining {
                return Err(remote_limit("bytes", limits.max_bytes));
            }
            bytes.extend_from_slice(&chunk[..count]);
        }
        Ok((bytes, charset))
    }

    fn fetch_text(
        path: &str,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<String, DataLoadError> {
        if is_url(path) {
            let (bytes, charset) = Self::fetch_remote_bytes(path, limits)?;
            let encoding = charset
                .as_deref()
                .and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
                .unwrap_or(encoding_rs::UTF_8);
            let (text, _, _) = encoding.decode(&bytes);
            Ok(text.into_owned())
        } else {
            fs::read_to_string(path).map_err(|err| DataLoadError::Io(err.into()))
        }
    }

    fn load_json_rows(
        path: &str,
        lines: bool,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let data = Self::fetch_text(path, limits)?;
        let remote = is_url(path);

        if lines {
            let mut rows = Vec::new();
            for (idx, line) in data.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                if remote && rows.len() >= limits.max_rows {
                    return Err(remote_limit("rows", limits.max_rows));
                }
                let value: serde_json::Value =
                    serde_json::from_str(line).map_err(|err| DataLoadError::Json(anyhow!(err)))?;
                rows.push(row_from_json_value(value, idx + 1)?);
            }
            debug!(rows = rows.len(), "jsonl rows loaded");
            return Ok(rows);
        }

        let rows = if remote && data.trim_start().starts_with('[') {
            let limit_hit = Cell::new(false);
            let mut deserializer = serde_json::Deserializer::from_str(&data);
            let items = LimitedJsonArray {
                max_rows: limits.max_rows,
                limit_hit: &limit_hit,
            }
            .deserialize(&mut deserializer)
            .map_err(|err| {
                if limit_hit.get() {
                    remote_limit("rows", limits.max_rows)
                } else {
                    DataLoadError::Json(anyhow!(err))
                }
            })?;
            deserializer.end().map_err(|err| DataLoadError::Json(anyhow!(err)))?;
            items
                .into_iter()
                .enumerate()
                .map(|(idx, item)| row_from_json_value(item, idx + 1))
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let value: serde_json::Value =
                serde_json::from_str(&data).map_err(|err| DataLoadError::Json(anyhow!(err)))?;
            match value {
                serde_json::Value::Array(items) => {
                items
                    .into_iter()
                    .enumerate()
                    .map(|(idx, item)| row_from_json_value(item, idx + 1))
                    .collect::<std::result::Result<Vec<_>, _>>()?
                }
                other => {
                    if remote && limits.max_rows == 0 {
                        return Err(remote_limit("rows", limits.max_rows));
                    }
                    vec![row_from_json_value(other, 1)?]
                }
            }
        };

        debug!(rows = rows.len(), "json rows loaded");
        Ok(rows)
    }

    #[cfg(feature = "data")]
    fn load_csv_rows(
        path: &str,
        delimiter: char,
        has_headers: bool,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        if is_url(path) {
            let (bytes, _) = Self::fetch_remote_bytes(path, limits).map_err(|err| match err {
                DataLoadError::Io(source) => DataLoadError::Csv(source),
                other => other,
            })?;

            let cursor = Cursor::new(bytes);
            let mut reader = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_reader(cursor);
            return Self::collect_csv_rows(&mut reader, has_headers, Some(limits.max_rows));
        }

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter as u8)
            .has_headers(has_headers)
            .from_path(path)
            .map_err(|err| DataLoadError::Csv(err.into()))?;
        Self::collect_csv_rows(&mut reader, has_headers, None)
    }

    #[cfg(feature = "data")]
    fn collect_csv_rows<R: std::io::Read>(
        reader: &mut csv::Reader<R>,
        has_headers: bool,
        max_rows: Option<usize>,
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

        let mut rows = Vec::new();
        for (idx, record) in reader.records().enumerate() {
            if let Some(limit) = max_rows
                && rows.len() >= limit
            {
                return Err(remote_limit("rows", limit));
            }
            let record = record.map_err(|err| DataLoadError::Csv(err.into()))?;
            rows.push(csv_record_to_row_record(
                &record,
                idx + 1,
                header_names.as_deref(),
            ));
        }

        debug!(rows = rows.len(), "csv rows loaded");
        Ok(rows)
    }

    #[cfg(feature = "data")]
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

    #[cfg(feature = "data")]
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

    #[cfg(feature = "data")]
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
                Self::load_json_rows(
                    path_str,
                    file.ends_with(".jsonl"),
                    RemoteLoadLimits::default(),
                )?
            } else {
                Self::load_csv_rows(path_str, ',', true, RemoteLoadLimits::default())?
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

/// Enumerates a row struct's (field name, declared type) pairs from its Facet shape.
fn row_struct_fields<E: for<'f> Facet<'f>>()
-> std::result::Result<Vec<(&'static str, FieldType)>, DataLoadError> {
    let shape = E::SHAPE;
    let struct_type = match &shape.ty {
        Type::User(UserType::Struct(struct_type)) => struct_type,
        _ => {
            return Err(DataLoadError::Json(anyhow!(
                "row type `{}` must be a struct",
                shape.type_identifier
            )));
        }
    };
    Ok(struct_type
        .fields
        .iter()
        .filter(|field| !field.should_skip_deserializing())
        .map(|field| (field.name, field_type_from_shape(field.shape())))
        .collect())
}

fn struct_from_row<E: DeserializeOwned>(
    row: &RowRecord,
    fields: &[(&'static str, FieldType)],
    opts: &TypedLoadOptions,
) -> std::result::Result<E, DataLoadError> {
    let mut used_source_fields = HashSet::new();
    let mut map = Map::new();

    for (name, type_ir) in fields {
        let source_field = resolve_source_field(name, opts);
        match row.values.get(source_field) {
            Some(value) => {
                map.insert(name.to_string(), coerce_row_value(value, type_ir));
                used_source_fields.insert(source_field.to_string());
            }
            // Missing optional fields deserialize as `None`; anything else is an error.
            None if type_ir.is_optional() => {}
            None => {
                return Err(DataLoadError::MissingField {
                    row: row.row_index,
                    field: name.to_string(),
                });
            }
        }
    }

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

    serde_json::from_value::<E>(Value::Object(map)).map_err(|err| DataLoadError::TypeMismatch {
        row: row.row_index,
        field: "row".to_string(),
        message: err.to_string(),
    })
}

/// Nudges a loosely-typed source value (common with CSV, where every cell starts as text)
/// toward the field's declared type before strict serde deserialization. This preserves the
/// tolerant loading behavior the vendored BAML stack used to provide: a `"4"` cell fills a
/// `String` field as `"4"` and an `i64` field as `4`. Values that don't cleanly convert are
/// passed through unchanged so serde surfaces a precise type-mismatch error.
fn coerce_row_value(value: &Value, field_type: &FieldType) -> Value {
    match field_type.peel() {
        FieldType::String => match value {
            Value::String(_) => value.clone(),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Bool(flag) => Value::String(flag.to_string()),
            _ => value.clone(),
        },
        FieldType::Int => match value {
            Value::String(text) => text
                .trim()
                .parse::<i64>()
                .map(|parsed| Value::Number(parsed.into()))
                .unwrap_or_else(|_| value.clone()),
            _ => value.clone(),
        },
        FieldType::Float => match value {
            Value::String(text) => text
                .trim()
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number)
                .unwrap_or_else(|| value.clone()),
            _ => value.clone(),
        },
        FieldType::Bool => match value {
            Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => value.clone(),
            },
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

fn remote_limit(resource: &str, limit: usize) -> DataLoadError {
    DataLoadError::Io(anyhow!("remote dataset exceeded {resource} limit ({limit})"))
}

fn remote_timeout(timeout: Duration) -> DataLoadError {
    DataLoadError::Io(anyhow!("remote dataset timed out after {timeout:?}"))
}

struct LimitedJsonArray<'a> {
    max_rows: usize,
    limit_hit: &'a Cell<bool>,
}

impl<'de> DeserializeSeed<'de> for LimitedJsonArray<'_> {
    type Value = Vec<serde_json::Value>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for LimitedJsonArray<'_> {
    type Value = Vec<serde_json::Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON array")
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while items.len() < self.max_rows {
            match seq.next_element()? {
                Some(item) => items.push(item),
                None => return Ok(items),
            }
        }
        if seq.next_element::<IgnoredAny>()?.is_some() {
            self.limit_hit.set(true);
            return Err(A::Error::custom("remote row limit exceeded"));
        }
        Ok(items)
    }
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

#[cfg(feature = "data")]
fn parse_csv_cell(cell: &str) -> serde_json::Value {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return serde_json::Value::String(String::new());
    }

    serde_json::from_str::<serde_json::Value>(trimmed)
        .unwrap_or_else(|_| serde_json::Value::String(cell.to_string()))
}

#[cfg(feature = "data")]
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

#[cfg(feature = "data")]
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
