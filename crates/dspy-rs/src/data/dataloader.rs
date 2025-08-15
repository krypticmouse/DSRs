use crate::data::example::Example;

pub trait DataLoader {
    fn load_json(
        &self,
        path: &str,
        lines: bool,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        compression: &str,
    ) -> Vec<Example> {
        todo!()
    }

    fn save_json(&self, path: &str, examples: Vec<Example>, lines: bool, compression: &str) {
        todo!()
    }

    
    fn load_csv(
        &self,
        path: &str,
        delimiter: char,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        compression: &str,
    ) -> Vec<Example> {
        todo!()
    }

    fn save_csv(&self, path: &str, examples: Vec<Example>, delimiter: char) {
        todo!()
    }


    fn load_parquet(
        &self,
        path: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
        compression: &str,
    ) -> Vec<Example> {
        todo!()
    }

    fn save_parquet(&self, path: &str, examples: Vec<Example>) {
        todo!()
    }
}
