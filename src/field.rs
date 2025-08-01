pub trait Field {
    fn desc(&self) -> &str;
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct In {
    pub desc: String,
}

impl Field for In {
    fn desc(&self) -> &str {
        &self.desc
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct Out {
    pub desc: String,
}

impl Field for Out {
    fn desc(&self) -> &str {
        &self.desc
    }
}
