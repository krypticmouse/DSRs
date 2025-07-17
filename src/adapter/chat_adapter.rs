use crate::adapter::base::Adapter;
use crate::signature::signature::Signature;
use crate::signature::field::Field;
use std::collections::HashMap;

fn get_type_name<T>(_ : &T) -> &'static str {
    std::any::type_name::<T>()
}

trait ChatAdapter: Adapter {
    fn format_system_message(&self, signature: Signature) -> String {
        let field_description = self.format_field_description(signature);
        let field_structure = self.format_field_structure(signature);
        let task_description = self.format_task_description(signature);

        format!("{field_description}\n{field_structure}\n{task_description}")
    }

    fn get_field_attribute_list(&self, field_iter: Iterator<(&String, &Field)>) -> String {
        let mut field_attributes = String::new();
        for (i, (field_name, field)) in field_iter.enumerate() {
            field_attributes.push_str(format!("{}. `{}` ({})\n", i, field_name, get_type_name(field)).as_str());
        }
        field_attributes
    }

    fn format_field_description(&self, signature: Signature) -> String {
        let input_field_description = self.get_field_attribute_list(signature.input_fields.iter());
        let output_field_description = self.get_field_attribute_list(signature.output_fields.iter());

        format!("Your input fields are:\n{input_field_description}\n\nYour output fields are:\n{output_field_description}")
    }

    fn get_field_structure(&self, field_iter: Iterator<(&String, &Field)>) -> String {
        let mut field_structure = String::new();
        for (field_name, _) in field_iter {
            field_structure.push_str(format!("[[ ## {field_name} ## ]]\n{field_name}\n\n").as_str());
        }
        field_structure
    }

    fn format_field_structure(&self, signature: Signature) -> String {
        let input_field_structure = self.get_field_structure(signature.input_fields.iter());
        let output_field_structure = self.get_field_structure(signature.output_fields.iter());

        format!("All interactions will be structured in the following way, with the appropriate values filled in.\n\n{input_field_structure}\n\n{output_field_structure}\n\n[[ ## completed ## ]]")
    }

    fn format_task_description(&self, signature: Signature) -> String {
        let instruction = if !signature.instruction.is_empty() {
            format!(
                "Given the fields `{}`, produce the fields `{}`.", 
                signature.input_fields.keys().join(", "), 
                signature.output_fields.keys().join(", ")
            )
        } else {
            signature.instruction.clone()
        };

        format!("In adhering to this structure, your objective is: {instruction}")
    }

    fn format_user_message(&self, signature: Signature, inputs: HashMap<String, String>) -> String {
        let mut input_str = String::new();
        for (field_name, field_value) in inputs {
            input_str.push_str(format!("[[ ## {field_name} ## ]]\n{field_value}\n\n").as_str());
        }

        let mut user_message = format!("Respond with the corresponding output fields, starting with the field `{}`,", signature.output_fields.keys()[0]);
        for (field_name, _) in signature.output_fields.iter().skip(1) {
            user_message.push_str(format!(" then `{}`,", field_name).as_str());
        }
        user_message.push_str("and then ending with the marker for `completed`.");

        format!("{input_str}\n\n{user_message}")
    }

    fn parse_response(&self, signature: Signature, response: String) -> HashMap<String, String> {
        let mut output = HashMap::new();

        for (field_name, _) in signature.output_fields.iter() {
            let field_value = response.split(format!("[[ ## {field_name} ## ]]\n").as_str()).nth(1).unwrap();
            output.insert(field_name.clone(), field_value.split("[[ ## ").nth(0).unwrap().trim().to_string());
        }

        output
    }
}