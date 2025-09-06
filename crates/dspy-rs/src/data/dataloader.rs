use anyhow::Result;
use polars::prelude::*;
use serde_json::Value;
use std::fs::File;

use crate::Example;

/// Convert a Vec of Examples to a Polars DataFrame
pub fn examples_to_dataframe(examples: Vec<Example>) -> Result<DataFrame> {
    if examples.is_empty() {
        return Ok(DataFrame::empty());
    }

    // Get all unique keys from all examples
    let mut all_keys = std::collections::HashSet::new();
    for example in &examples {
        for key in example.keys() {
            all_keys.insert(key);
        }
    }

    // Create series for each key
    let mut columns = Vec::new();

    for key in all_keys {
        let mut values: Vec<String> = Vec::new();

        for example in &examples {
            let value = example.get(&key, Some(""));
            // Convert Value to String
            let str_value = match value {
                Value::String(s) => s,
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => String::new(),
                _ => value.to_string(),
            };
            values.push(str_value);
        }

        let series = Series::new(key.as_str().into(), values);
        columns.push(series.into());
    }

    Ok(DataFrame::new(columns)?)
}

fn dataframe_to_examples(
    df: DataFrame,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
) -> Vec<Example> {
    df.iter()
        .map(|row| {
            Example::new(
                row.iter()
                    .map(|cell| (cell.to_string(), cell.to_string().into()))
                    .collect(),
                input_keys.clone(),
                output_keys.clone(),
            )
        })
        .collect()
}

pub trait DataLoader {
    fn load_json(
        &self,
        path: &str,
        lines: bool,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        n_threads: Option<usize>,
    ) -> Result<Vec<Example>> {
        let file = File::open(path)?;
        let data = if lines {
            JsonReader::new(file).finish()?
        } else {
            JsonLineReader::new(file)
                .with_n_threads(n_threads)
                .finish()?
        };
        let examples = dataframe_to_examples(data, input_keys, output_keys);
        Ok(examples)
    }

    fn save_json(&self, path: &str, examples: Vec<Example>, lines: bool) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        let mut df = examples_to_dataframe(examples)?;
        JsonWriter::new(&mut file)
            .with_json_format(if lines {
                JsonFormat::JsonLines
            } else {
                JsonFormat::Json
            })
            .finish(&mut df)?;
        Ok(())
    }

    fn load_csv(
        &self,
        path: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        n_threads: Option<usize>,
    ) -> Result<Vec<Example>> {
        let examples = CsvReadOptions::default()
            .with_n_threads(n_threads)
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?
            .iter()
            .map(|row| {
                let data = row
                    .iter()
                    .map(|cell| (cell.to_string(), cell.to_string().into()))
                    .collect();

                Example::new(data, input_keys.clone(), output_keys.clone())
            })
            .collect::<Vec<Example>>();
        Ok(examples)
    }

    fn save_csv(&self, path: &str, examples: Vec<Example>, delimiter: char) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        let mut df = examples_to_dataframe(examples)?;
        CsvWriter::new(&mut file)
            .with_separator(delimiter as u8)
            .finish(&mut df)?;
        Ok(())
    }

    fn load_parquet(
        &self,
        path: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Result<Vec<Example>> {
        let file = File::open(path)?;
        let data = ParquetReader::new(file).finish()?;
        let examples = dataframe_to_examples(data, input_keys, output_keys);
        Ok(examples)
    }

    fn save_parquet(&self, path: &str, examples: Vec<Example>) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        let mut df = examples_to_dataframe(examples)?;
        ParquetWriter::new(&mut file).finish(&mut df)?;
        Ok(())
    }
}
