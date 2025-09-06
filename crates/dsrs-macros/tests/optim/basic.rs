use dspy_rs::core::module::Optimizable;
use dsrs_macros::Optimizable;
use dspy_rs::{signature, Predict};
use std::collections::HashMap;

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
        first: Predict::new(signature! {
            "question": String -> "answer": String,
        }),
        second: Predict::new(signature! {
            "question": String -> "answer": String,
        }),
        random: 42,
    };

    // Option 1: Print parameter names only
    println!("Parameter names:");
    for (name, _param) in module.parameters() {
        println!("  - {}", name);
    }

    // Option 2: Print parameter names with memory addresses
    println!("\nParameters with addresses:");
    for (name, param) in module.parameters() {
        println!("  - {}: {:p}", name, param as *const _);
    }

    // Option 3: Print parameter names with type information
    println!("\nParameters with type info:");
    for (name, param) in module.parameters() {
        // Using std::any to get type name (requires importing)
        let type_name = param.signature.instruction();
        println!("  - {} (type: {})", name, type_name);
    }
}
