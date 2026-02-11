use crate::data::example::Example;
use csv::StringRecord;

use regex::Regex;
use std::sync::LazyLock;

#[allow(dead_code)]
static IS_URL_PAT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new("((http|https)://)(www.)?[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)"
).unwrap()
});

/// Converts a CSV [`StringRecord`] into a [`RawExample`](crate::RawExample).
///
/// If `field_names` is provided and matches the record length, uses those as keys.
/// Otherwise falls back to `column_0`, `column_1`, etc.
pub fn string_record_to_example(
    record: StringRecord,
    field_names: Option<&[String]>,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
) -> Example {
    let pairs = if let Some(names) = field_names {
        if names.len() == record.len() {
            names
                .iter()
                .zip(record.iter())
                .map(|(name, cell)| (name.clone(), cell.to_string().into()))
                .collect()
        } else {
            record
                .iter()
                .enumerate()
                .map(|(idx, cell)| (format!("column_{idx}"), cell.to_string().into()))
                .collect()
        }
    } else {
        record
            .iter()
            .enumerate()
            .map(|(idx, cell)| (format!("column_{idx}"), cell.to_string().into()))
            .collect()
    };

    Example::new(
        pairs,
        input_keys.clone(),
        output_keys.clone(),
    )
}

/// Returns `true` if the string looks like an HTTP(S) URL.
pub fn is_url(path: &str) -> bool {
    IS_URL_PAT.is_match(path)
}
