use std::fs::File;
use rayon::prelude::*;
use std::io::{BufReader, BufRead, BufWriter, Write};

use crate::data::example::Example;

pub fn load_jsonl(path: &str, input_keys: Vec<String>, output_key: Vec<String>) -> Vec<Example> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    
    let lines: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .collect();
    
    let examples: Vec<Example> = lines
        .par_iter()
        .map(|line| {
            let mut example: Example = serde_json::from_str(line).unwrap();
            example.input_keys = input_keys.clone();
            example.output_keys = output_key.clone();
            example
        })
        .collect();
    
    examples
}

pub fn save_examples_as_jsonl(path: &str, examples: Vec<Example>) {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    
    for example in examples {
        let json = serde_json::to_string(&example).unwrap();

        writer.write_all(json.as_bytes()).unwrap();
        writer.write_all(b"\n").unwrap();
    }
}
