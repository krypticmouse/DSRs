use anyhow::Result;
use arrow::array::{Array, StringArray};
use csv::{ReaderBuilder, WriterBuilder};
use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use reqwest;
use std::fs;
use std::io::Cursor;
use std::{collections::HashMap, path::Path};
use tracing::{Span, debug};

use crate::{RawExample, is_url, string_record_to_example};

/// Loads datasets from JSON, CSV, Parquet files, and HuggingFace Hub.
///
/// All methods return `Vec<RawExample>` — untyped key-value pairs. Specify
/// `input_keys` and `output_keys` to tell the system which fields are inputs
/// vs outputs (this metadata flows through to typed conversion later).
///
/// Supports both local file paths and HTTP(S) URLs for JSON and CSV.
///
/// ```ignore
/// let examples = DataLoader::load_json(
///     "data/hotpotqa.jsonl",
///     true,  // JSON lines format
///     vec!["question".into()],
///     vec!["answer".into()],
/// )?;
/// ```
pub struct DataLoader;

impl DataLoader {
    #[tracing::instrument(
        name = "dsrs.data.load_json",
        level = "debug",
        skip(input_keys, output_keys),
        fields(
            is_url = is_url(path),
            input_keys = input_keys.len(),
            output_keys = output_keys.len()
        )
    )]
    /// Loads examples from a JSON file or URL.
    ///
    /// When `lines` is `true`, treats each line as a separate JSON object (JSONL format).
    /// When `false`, expects a single JSON object.
    ///
    /// # Errors
    ///
    /// - Network error if `path` is a URL and the request fails
    /// - Parse error if the JSON is malformed
    pub fn load_json(
        path: &str,
        lines: bool,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Result<Vec<RawExample>> {
        let source_is_url = is_url(path);
        let data = if source_is_url {
            let response = reqwest::blocking::get(path)?;
            response.text()?
        } else {
            fs::read_to_string(path)?
        };

        let examples: Vec<RawExample> = if lines {
            let lines = data.lines().collect::<Vec<&str>>();
            let span = Span::current();

            lines
                .par_iter()
                .map(|line| {
                    let span = span.clone();
                    span.in_scope(|| {
                        RawExample::new(
                            serde_json::from_str(line).unwrap(),
                            input_keys.clone(),
                            output_keys.clone(),
                        )
                    })
                })
                .collect()
        } else {
            vec![RawExample::new(
                serde_json::from_str(&data).unwrap(),
                input_keys.clone(),
                output_keys.clone(),
            )]
        };
        debug!(examples_loaded = examples.len(), "json examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.save_json",
        level = "debug",
        skip(examples),
        fields(examples = examples.len())
    )]
    /// Saves examples to a JSON file.
    ///
    /// When `lines` is `true`, writes one JSON object per line (JSONL format).
    pub fn save_json(path: &str, examples: Vec<RawExample>, lines: bool) -> Result<()> {
        let data = if lines {
            examples
                .into_iter()
                .map(|example| serde_json::to_string(&example).unwrap())
                .collect::<Vec<String>>()
                .join("\n")
        } else {
            serde_json::to_string(&examples).unwrap()
        };
        fs::write(path, data)?;
        debug!("json examples saved");
        Ok(())
    }

    #[tracing::instrument(
        name = "dsrs.data.load_csv",
        level = "debug",
        skip(input_keys, output_keys),
        fields(
            is_url = is_url(path),
            input_keys = input_keys.len(),
            output_keys = output_keys.len()
        )
    )]
    /// Loads examples from a CSV file or URL.
    ///
    /// When `has_headers` is `true`, uses the first row as field names. When `false`,
    /// uses `input_keys` and `output_keys` as field names (falling back to `column_0`,
    /// `column_1`, etc. if those are also empty).
    ///
    /// # Errors
    ///
    /// - Network error if `path` is a URL and the request fails
    /// - Parse error if the CSV is malformed
    pub fn load_csv(
        path: &str,
        delimiter: char,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        has_headers: bool,
    ) -> Result<Vec<RawExample>> {
        let mut fallback_field_names = input_keys.clone();
        fallback_field_names.extend(output_keys.clone());

        let source_is_url = is_url(path);
        let (records, header_field_names) = if source_is_url {
            let response = reqwest::blocking::get(path)?.bytes()?.to_vec();
            let cursor = Cursor::new(response);

            let mut reader = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_reader(cursor);

            let header_field_names = if has_headers {
                Some(
                    reader
                        .headers()?
                        .iter()
                        .map(|header| header.to_string())
                        .collect::<Vec<_>>(),
                )
            } else if !fallback_field_names.is_empty() {
                Some(fallback_field_names.clone())
            } else {
                None
            };

            let records: Vec<_> = reader.into_records().collect::<Result<Vec<_>, _>>()?;

            (records, header_field_names)
        } else {
            let mut reader = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_path(path)?;

            let header_field_names = if has_headers {
                Some(
                    reader
                        .headers()?
                        .iter()
                        .map(|header| header.to_string())
                        .collect::<Vec<_>>(),
                )
            } else if !fallback_field_names.is_empty() {
                Some(fallback_field_names.clone())
            } else {
                None
            };

            let records: Vec<_> = reader.into_records().collect::<Result<Vec<_>, _>>()?;

            (records, header_field_names)
        };
        let span = Span::current();
        let header_field_names = header_field_names.as_deref();

        let examples: Vec<RawExample> = records
            .par_iter()
            .map(|row| {
                let span = span.clone();
                span.in_scope(|| {
                    string_record_to_example(
                        row.clone(),
                        header_field_names,
                        input_keys.clone(),
                        output_keys.clone(),
                    )
                })
            })
            .collect();

        debug!(examples_loaded = examples.len(), "csv examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.save_csv",
        level = "debug",
        skip(examples),
        fields(examples = examples.len())
    )]
    /// Saves examples to a CSV file with the given delimiter.
    pub fn save_csv(path: &str, examples: Vec<RawExample>, delimiter: char) -> Result<()> {
        let mut writer = WriterBuilder::new()
            .delimiter(delimiter as u8)
            .from_path(path)?;
        let headers = examples[0].data.keys().cloned().collect::<Vec<String>>();
        writer.write_record(&headers)?;
        for example in examples {
            writer.write_record(
                example
                    .data
                    .values()
                    .map(|value| value.to_string())
                    .collect::<Vec<String>>(),
            )?;
        }
        debug!("csv examples saved");
        Ok(())
    }

    #[allow(clippy::while_let_on_iterator)]
    #[tracing::instrument(
        name = "dsrs.data.load_parquet",
        level = "debug",
        skip(input_keys, output_keys),
        fields(input_keys = input_keys.len(), output_keys = output_keys.len())
    )]
    /// Loads examples from a local Parquet file.
    ///
    /// Only reads string columns — other column types are silently skipped. Rows
    /// where all columns are null are skipped.
    ///
    /// Does not support URLs — use [`load_hf`](DataLoader::load_hf) for remote datasets.
    ///
    /// # Errors
    ///
    /// - File not found or I/O error
    /// - Invalid Parquet format
    pub fn load_parquet(
        path: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Result<Vec<RawExample>> {
        let file_path = Path::new(path);

        let file = fs::File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut record_batch_reader = builder.build()?;

        let mut examples = Vec::new();
        while let Some(record_batch_result) = record_batch_reader.next() {
            let record_batch = record_batch_result?;
            let schema = record_batch.schema();
            let num_rows = record_batch.num_rows();

            // Process each row
            for row_idx in 0..num_rows {
                let mut data = HashMap::new();

                for col_idx in 0..record_batch.num_columns() {
                    let column = record_batch.column(col_idx);
                    let column_name = schema.field(col_idx).name();

                    if let Some(string_array) = column.as_any().downcast_ref::<StringArray>()
                        && !string_array.is_null(row_idx)
                    {
                        let value = string_array.value(row_idx);
                        data.insert(column_name.to_string(), value.to_string().into());
                    }
                }

                if !data.is_empty() {
                    examples.push(RawExample::new(data, input_keys.clone(), output_keys.clone()));
                }
            }
        }
        debug!(examples_loaded = examples.len(), "parquet examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_hf",
        level = "debug",
        skip(input_keys, output_keys),
        fields(input_keys = input_keys.len(), output_keys = output_keys.len())
    )]
    /// Loads examples from a HuggingFace Hub dataset.
    ///
    /// Downloads and caches the dataset locally using `hf_hub`, then loads each
    /// file (Parquet, JSON, JSONL, or CSV) that matches the `subset` and `split`
    /// filters. Files are loaded in parallel via rayon.
    ///
    /// # Known issue: silent file errors
    ///
    /// Individual file load errors are silently swallowed (`.ok()`). If a Parquet
    /// file is corrupted or a JSON file is malformed, it's skipped without error and
    /// you get fewer examples than expected. Check `examples.len()` against your
    /// expectations. Set `verbose = true` to see which files are being loaded.
    ///
    /// # Errors
    ///
    /// - HuggingFace API error (auth, dataset not found)
    pub fn load_hf(
        dataset_id: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        subset: &str,
        split: &str,
        verbose: bool,
    ) -> Result<Vec<RawExample>> {
        let api = Api::new()?;
        let repo = api.dataset(dataset_id.to_string());

        // Get metadata and list of files using info()
        let metadata = repo.info()?;
        let files: Vec<&str> = metadata
            .siblings
            .iter()
            .map(|sib| sib.rfilename.as_str())
            .collect();
        debug!(files = files.len(), "hf dataset files discovered");
        let span = Span::current();

        let examples: Vec<_> = files
            .par_iter()
            .filter_map(|file: &&str| {
                let span = span.clone();
                span.in_scope(|| {
                    let extension = file.split(".").last().unwrap();
                    if !file.ends_with(".parquet")
                        && !extension.ends_with("json")
                        && !extension.ends_with("jsonl")
                        && !extension.ends_with("csv")
                    {
                        if verbose {
                            println!("Skipping file by extension: {file}");
                            debug!(file = *file, "skipping hf file by extension");
                        }
                        return None;
                    }

                    if (!subset.is_empty() && !file.contains(subset))
                        || (!split.is_empty() && !file.contains(split))
                    {
                        if verbose {
                            println!("Skipping file by subset or split: {file}");
                            debug!(file = *file, "skipping hf file by subset/split");
                        }
                        return None;
                    }

                    let file_path = repo.get(file).unwrap();
                    let os_str = file_path.as_os_str().to_str().unwrap();

                    if verbose {
                        println!("Loading file: {os_str}");
                        debug!(path = os_str, "loading hf file");
                    }

                    if os_str.ends_with(".parquet") {
                        DataLoader::load_parquet(os_str, input_keys.clone(), output_keys.clone())
                            .ok()
                    } else if os_str.ends_with(".json") || os_str.ends_with(".jsonl") {
                        let is_jsonl = os_str.ends_with(".jsonl");
                        DataLoader::load_json(
                            os_str,
                            is_jsonl,
                            input_keys.clone(),
                            output_keys.clone(),
                        )
                        .ok()
                    } else if os_str.ends_with(".csv") {
                        DataLoader::load_csv(
                            os_str,
                            ',',
                            input_keys.clone(),
                            output_keys.clone(),
                            true,
                        )
                        .ok()
                    } else {
                        None
                    }
                })
            })
            .flatten()
            .collect();

        if verbose {
            println!("Loaded {} examples", examples.len());
        }
        debug!(examples_loaded = examples.len(), "hf examples loaded");
        Ok(examples)
    }
}
