use anyhow::Result;
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use std::fs;

use crate::Example;

fn string_record_to_example(
    record: StringRecord,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
) -> Example {
    Example::new(
        record
            .iter()
            .map(|cell| (cell.to_string(), cell.to_string().into()))
            .collect(),
        input_keys.clone(),
        output_keys.clone(),
    )
}

pub trait DataLoader {
    fn load_json(
        &self,
        path: &str,
        lines: bool,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Result<Vec<Example>> {
        let data = fs::read_to_string(path)?;

        let examples: Vec<Example> = if lines {
            data.lines()
                .map(|line| {
                    let mut example: Example = serde_json::from_str(line).unwrap();
                    example.input_keys = input_keys.clone();
                    example.output_keys = output_keys.clone();
                    example
                })
                .collect()
        } else {
            serde_json::from_str(&data).unwrap()
        };
        Ok(examples)
    }

    fn save_json(&self, path: &str, examples: Vec<Example>, lines: bool) -> Result<()> {
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
        Ok(())
    }

    fn load_csv(
        &self,
        path: &str,
        delimiter: char,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        has_headers: bool,
    ) -> Result<Vec<Example>> {
        let df = ReaderBuilder::new()
            .delimiter(delimiter as u8)
            .has_headers(has_headers)
            .from_path(path)?
            .into_records();
        let examples = df
            .map(|row| {
                string_record_to_example(row.unwrap(), input_keys.clone(), output_keys.clone())
            })
            .collect();
        Ok(examples)
    }

    fn save_csv(&self, path: &str, examples: Vec<Example>, delimiter: char) -> Result<()> {
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
                    .cloned()
                    .map(|value| value.to_string())
                    .collect::<Vec<String>>(),
            )?;
        }
        Ok(())
    }
}
