use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field {
    In(String),
    Out(String),
}

impl Field {
    pub fn desc(&self) -> &str {
        match self {
            Field::In(desc) => desc,
            Field::Out(desc) => desc,
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Field::In(desc) => write!(f, "Input({desc})"),
            Field::Out(desc) => write!(f, "Output({desc})"),
        }
    }
}
