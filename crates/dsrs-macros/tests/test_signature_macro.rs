use dsrs_macros::Signature;

#[Signature(cot, hint)]
struct TestSignature {
    /// This is a test instruction
    /// What is the meaning of life?
    #[input(desc = "The main question to answer")]
    question: String,
    
    #[input(desc = "Additional context for the question")]  
    context: String,

    #[output(desc = "The answer to the question")]
    answer: i8,
    
    #[output(desc = "Confidence score")]
    confidence: f32,
}

#[allow(dead_code)]
struct TestOutput {
    output1: i8,
    output2: String,
    output3: bool,
}

#[Signature]
struct TestSignature2 {
    /// This is a test input
    ///
    /// What is the meaning of life?
    
    #[input(desc = "The first input")]
    input1: String,
    #[input(desc = "The second input")]
    input2: i8,
    #[output(desc = "The first output")]
    output1: TestOutput,
}

#[test]
fn test_signature_macro() {
    let signature = TestSignature::new();

    assert_eq!(signature.instruction, "This is a test instruction\nWhat is the meaning of life?");
    assert_eq!(signature.input_schema["question"]["type"], "String");
    assert_eq!(signature.input_schema["question"]["desc"], "The main question to answer");
    assert_eq!(signature.input_schema["context"]["type"], "String");
    assert_eq!(signature.input_schema["context"]["desc"], "Additional context for the question");
    assert_eq!(signature.output_schema["answer"]["type"], "i8");
    assert_eq!(signature.output_schema["answer"]["desc"], "The answer to the question");
    assert_eq!(signature.output_schema["reasoning"]["type"], "String");
    assert_eq!(signature.output_schema["reasoning"]["desc"], "Think step by step");
    assert_eq!(signature.output_schema["confidence"]["type"], "f32");
    assert_eq!(signature.output_schema["confidence"]["desc"], "Confidence score");
    assert_eq!(signature.input_schema["hint"]["type"], "String");
    assert_eq!(signature.input_schema["hint"]["desc"], "Hint for the query");

    let signature = TestSignature2::new();
    
    assert_eq!(signature.instruction, "This is a test input\n\nWhat is the meaning of life?");
    assert_eq!(signature.input_schema["input1"]["type"], "String");
    assert_eq!(signature.input_schema["input1"]["desc"], "The first input");
    assert_eq!(signature.input_schema["input2"]["type"], "i8");
    assert_eq!(signature.input_schema["input2"]["desc"], "The second input");
    assert_eq!(signature.output_schema["output1"]["type"], "TestOutput");
    assert_eq!(signature.output_schema["output1"]["desc"], "The first output");
}