use dspy_rs::core::module::Optimizable;
use dsrs_macros::Optimizable;

pub struct Predict {}

impl Optimizable for Predict {
    fn parameters(
        &mut self,
    ) -> std::collections::HashMap<std::string::String, &mut dyn Optimizable> {
        std::collections::HashMap::new()
    }
}

#[derive(Optimizable)]
pub struct MyModule {
    #[parameter]
    pub first: Predict,
    #[parameter]
    pub second: Predict,
    pub random: u32,
}

pub fn main() {
    let mut module = MyModule {
        first: Predict {},
        second: Predict {},
        random: 42,
    };

    module.parameters();
}
