use std::any::Any;
use std::fmt;

use smart_default::SmartDefault;

pub type FormatFn = fn(&dyn Any) -> String;

#[derive(Debug, Clone, PartialEq, Eq, SmartDefault)]
pub enum Field {
    #[default]
    InputField {
        #[default = ""]
        prefix: String,
        #[default = ""]
        desc: String,
        format: Option<FormatFn>,
        #[default = "String"]
        output_type: String,
    },
    OutputField {
        #[default = ""]
        prefix: String,
        #[default = ""]
        desc: String,
        format: Option<FormatFn>,
        #[default = "String"]
        output_type: String,
    },
}

impl Field {
    pub fn prefix(&self) -> &str {
        match self {
            Field::InputField { prefix, .. } => prefix,
            Field::OutputField { prefix, .. } => prefix,
        }
    }

    pub fn desc(&self) -> &str {
        match self {
            Field::InputField { desc, .. } => desc,
            Field::OutputField { desc, .. } => desc,
        }
    }

    pub fn format(&self) -> Option<&FormatFn> {
        match self {
            Field::InputField { format, .. } => format.as_ref(),
            Field::OutputField { format, .. } => format.as_ref(),
        }
    }

    pub fn output_type(&self) -> &str {
        match self {
            Field::InputField { output_type, .. } => output_type,
            Field::OutputField { output_type, .. } => output_type,
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Field::InputField {
                prefix,
                desc,
                format,
                output_type: _,
            } => write!(
                f,
                "InputField(\n\tprefix: {},\n\tdesc: {},\n\tformat: {:?}\n)",
                prefix,
                desc,
                format.is_some()
            ),
            Field::OutputField {
                prefix,
                desc,
                format,
                output_type: _,
            } => write!(
                f,
                "OutputField(\n\tprefix: {},\n\tdesc: {},\n\tformat: {:?}\n)",
                prefix,
                desc,
                format.is_some()
            ),
        }
    }
}
