use crate::adapter::base::Adapter;
use crate::data::{
    example::Example,
    prediction::{LmUsage, Prediction},
};
use crate::field::Field;
use crate::signature::Signature;
use std::collections::HashMap;

use indexmap::IndexMap;
use openrouter_rs::types::CompletionsResponse;

#[derive(Default, Clone)]
pub struct ChatAdapter;

impl ChatAdapter {
    fn get_field_attribute_list<T: Field>(&self, field_iter: &IndexMap<String, T>) -> String {
        let mut field_attributes = String::new();
        for (i, (field_name, field)) in field_iter.iter().enumerate() {
            field_attributes.push_str(format!("{}. `{field_name}`", i + 1).as_str());
            if !field.desc().is_empty() {
                field_attributes.push_str(format!(": {}", field.desc()).as_str());
            }
            field_attributes.push('\n');
        }
        field_attributes
    }

    fn get_field_structure<T: Field>(&self, field_iter: &IndexMap<String, T>) -> String {
        let mut field_structure = String::new();
        for (field_name, _) in field_iter {
            field_structure
                .push_str(format!("[[ ## {field_name} ## ]]\n{field_name}\n\n").as_str());
        }
        field_structure
    }
}

impl Adapter for ChatAdapter {
    fn format_system_message(&self, signature: Signature) -> String {
        let field_description = self.format_field_description(signature.clone());
        let field_structure = self.format_field_structure(signature.clone());
        let task_description = self.format_task_description(signature.clone());

        format!("{field_description}\n{field_structure}\n{task_description}")
    }

    fn format_field_description(&self, signature: Signature) -> String {
        let input_field_description = self.get_field_attribute_list(&signature.input_fields);
        let output_field_description = self.get_field_attribute_list(&signature.output_fields);

        format!(
            "Your input fields are:\n{input_field_description}\nYour output fields are:\n{output_field_description}"
        )
    }

    fn format_field_structure(&self, signature: Signature) -> String {
        let input_field_structure = self.get_field_structure(&signature.input_fields);
        let output_field_structure = self.get_field_structure(&signature.output_fields);

        format!(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n{input_field_structure}{output_field_structure}[[ ## completed ## ]]\n"
        )
    }

    fn format_task_description(&self, signature: Signature) -> String {
        let instruction = if signature.instruction.is_empty() {
            format!(
                "Given the fields `{}`, produce the fields `{}`.",
                signature
                    .input_fields
                    .keys()
                    .map(|k| k.as_str())
                    .collect::<Vec<&str>>()
                    .join(", "),
                signature
                    .output_fields
                    .keys()
                    .map(|k| k.as_str())
                    .collect::<Vec<&str>>()
                    .join(", ")
            )
        } else {
            signature.instruction.clone()
        };

        format!("In adhering to this structure, your objective is:\n\t{instruction}")
    }

    fn format_user_message(&self, signature: Signature, inputs: Example) -> String {
        let mut input_str = String::new();
        for (field_name, _) in signature.input_fields.iter() {
            input_str.push_str(
                format!(
                    "[[ ## {field_name} ## ]]\n{field_value}\n\n",
                    field_name = field_name,
                    field_value = inputs.get(field_name, None)
                )
                .as_str(),
            );
        }

        let mut user_message = format!(
            "Respond with the corresponding output fields, starting with the field `{}`,",
            signature.output_fields.keys()[0]
        );
        for (field_name, _) in signature.output_fields.iter().skip(1) {
            user_message.push_str(format!(" then `{field_name}`,").as_str());
        }
        user_message.push_str(" and then ending with the marker for `completed`.");

        format!("{input_str}{user_message}")
    }

    fn parse_response(&self, signature: Signature, response: CompletionsResponse) -> Prediction {
        let mut output = HashMap::new();

        let response_content = if let openrouter_rs::types::Choice::NonStreaming(non_streaming) =
            &response.choices[0]
        {
            non_streaming.message.content.as_ref().unwrap()
        } else {
            panic!("Expected non-streaming choice");
        };

        for (field_name, _) in signature.output_fields.iter() {
            let field_value = response_content
                .split(format!("[[ ## {field_name} ## ]]\n").as_str())
                .nth(1)
                .unwrap();
            output.insert(
                field_name.clone(),
                field_value
                    .split("[[ ## ")
                    .nth(0)
                    .unwrap()
                    .trim()
                    .to_string(),
            );
        }

        Prediction {
            data: output,
            lm_usage: LmUsage::default(),
        }
    }
}
