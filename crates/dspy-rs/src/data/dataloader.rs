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

use crate::{Example, is_url, string_record_to_example};

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
    pub fn load_json(
        path: &str,
        lines: bool,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Result<Vec<Example>> {
        let source_is_url = is_url(path);
        let data = if source_is_url {
            let response = reqwest::blocking::get(path)?;
            response.text()?
        } else {
            fs::read_to_string(path)?
        };

        let examples: Vec<Example> = if lines {
            let lines = data.lines().collect::<Vec<&str>>();
            let span = Span::current();

            lines
                .par_iter()
                .map(|line| {
                    let span = span.clone();
                    span.in_scope(|| {
                        Example::new(
                            serde_json::from_str(line).unwrap(),
                            input_keys.clone(),
                            output_keys.clone(),
                        )
                    })
                })
                .collect()
        } else {
            vec![Example::new(
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
    pub fn save_json(path: &str, examples: Vec<Example>, lines: bool) -> Result<()> {
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
    pub fn load_csv(
        path: &str,
        delimiter: char,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        has_headers: bool,
    ) -> Result<Vec<Example>> {
        let source_is_url = is_url(path);
        let records = if source_is_url {
            let response = reqwest::blocking::get(path)?.bytes()?.to_vec();
            let cursor = Cursor::new(response);

            let records: Vec<_> = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_reader(cursor)
                .into_records()
                .collect::<Result<Vec<_>, _>>()?;

            records
        } else {
            let records: Vec<_> = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_path(path)?
                .into_records()
                .collect::<Result<Vec<_>, _>>()?;

            records
        };
        let span = Span::current();

        let examples: Vec<Example> = records
            .par_iter()
            .map(|row| {
                let span = span.clone();
                span.in_scope(|| {
                    string_record_to_example(row.clone(), input_keys.clone(), output_keys.clone())
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
    pub fn save_csv(path: &str, examples: Vec<Example>, delimiter: char) -> Result<()> {
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
    pub fn load_parquet(
        path: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Result<Vec<Example>> {
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
                    examples.push(Example::new(data, input_keys.clone(), output_keys.clone()));
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
    pub fn load_hf(
        dataset_id: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        subset: &str,
        split: &str,
        verbose: bool,
    ) -> Result<Vec<Example>> {
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
