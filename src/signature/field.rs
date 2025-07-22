use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field<'a> {
    In(&'a str),
    Out(&'a str),
}

impl<'a> Field<'a> {
    pub fn desc(&self) -> &'a str {
        match self {
            Field::In(desc) => desc,
            Field::Out(desc) => desc,
        }
    }
}

impl<'a> fmt::Display for Field<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Field::In(desc) => write!(f, "Input({desc})"),
            Field::Out(desc) => write!(f, "Output({desc})"),
        }
    }
}
