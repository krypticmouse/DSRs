use schemars::JsonSchema;
use std::sync::Arc;
use tokio::sync::Mutex;

use dspy_rs::{
    Cache, Chat, ChatAdapter, DummyLM, Example, LM, LMConfig, Message, MetaSignature, Prediction,
    Signature, adapter::Adapter, hashmap, sign,
};

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter() {
    let signature = sign! {
        (problem: String) -> answer: String
    };

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        Example::new(
            hashmap! {
                "problem".to_string() => "What is the capital of France?".to_string().into(),
                "answer".to_string() => "Paris".to_string().into(),
            },
            vec!["problem".to_string()],
            vec!["answer".to_string()],
        ),
    );

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    assert_eq!(
        json[0]["content"],
        "Your input fields are:\n1. `problem` (String)\n\nYour output fields are:\n1. `answer` (String)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tGiven the fields `problem`, produce the fields `answer`."
    );
    assert_eq!(
        json[1]["content"],
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\nRespond with the corresponding output fields, starting with the field `answer`, and then ending with the marker for `completed`.".to_string()
    );

    let response = lm
        .call(
            Chat::new(vec![
                Message::system("You are a helpful assistant."),
                Message::user("Hello, world!"),
            ]),
            "test",
            "[[ ## answer ## ]]\n150 degrees\n\n[[ ## completed ## ]]".to_string(),
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response.0);

    assert_eq!(output.len(), 1);
    assert_eq!(output.get("answer").unwrap(), "150 degrees");
}

#[allow(dead_code)]
#[Signature(cot, hint)]
struct TestSignature {
    ///You are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer.

    #[input]
    pub problem: String,
    #[output]
    pub answer: String,
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_multiple_fields() {
    let signature = TestSignature::new();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        Example::new(
            hashmap! {
                "problem".to_string() => "What is the capital of France?".to_string().into(),
                "hint".to_string() => "The capital of France is Paris.".to_string().into(),
            },
            vec!["problem".to_string(), "hint".to_string()],
            vec!["reasoning".to_string(), "answer".to_string()],
        ),
    );

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    assert_eq!(
        json[0]["content"],
        "Your input fields are:\n1. `problem` (String)\n2. `hint` (String): Hint for the query\n\nYour output fields are:\n1. `reasoning` (String): Think step by step\n2. `answer` (String)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## hint ## ]]\nhint\n\n[[ ## reasoning ## ]]\nreasoning\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tYou are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer.".to_string()
    );
    assert_eq!(
        json[1]["content"],
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\n[[ ## hint ## ]]\nThe capital of France is Paris.\n\nRespond with the corresponding output fields, starting with the field `reasoning`, then `answer`, and then ending with the marker for `completed`."
    );

    let response = lm
        .call(
            Chat::new(vec![
                Message::system("You are a helpful assistant."),
                Message::user("Hello, world!"),
            ]),
            "test",
            "[[ ## reasoning ## ]]\nThe capital of France is Paris.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]".to_string(),
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response.0);

    assert_eq!(output.len(), 2);
    assert_eq!(
        output.get("reasoning").unwrap(),
        "The capital of France is Paris."
    );
    assert_eq!(output.get("answer").unwrap(), "Paris");
}

#[allow(dead_code)]
#[derive(JsonSchema)]
struct TestOutput {
    pub reasoning: String,
    pub rating: i8,
}

#[allow(dead_code)]
#[Signature]
struct TestSignature2 {
    #[input]
    pub problem: String,
    #[input]
    pub hint: i8,
    #[output]
    pub output: TestOutput,
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_multiple_fields_and_output_schema() {
    let signature = TestSignature2::new();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        Example::new(
            hashmap! {
                "problem".to_string() => "What is the capital of France?".to_string().into(),
                "hint".to_string() => "The capital of France is Paris.".to_string().into(),
            },
            vec!["problem".to_string(), "hint".to_string()],
            vec!["output".to_string()],
        ),
    );

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    assert_eq!(
        json[0]["content"],
        "Your input fields are:\n1. `problem` (String)\n2. `hint` (i8)\n\nYour output fields are:\n1. `output` (TestOutput)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## hint ## ]]\nhint\t# note: the value you produce must be a single i8 value\n\n[[ ## output ## ]]\noutput\t# note: the value you produce must adhere to the JSON schema: {\"reasoning\":{\"type\":\"string\"},\"rating\":{\"type\":\"integer\",\"format\":\"int8\",\"minimum\":-128,\"maximum\":127}}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tGiven the fields `problem`, `hint`, produce the fields `output`.".to_string()
    );
    assert_eq!(
        json[1]["content"],
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\n[[ ## hint ## ]]\nThe capital of France is Paris.\n\nRespond with the corresponding output fields, starting with the field `output` (must be formatted as valid Rust TestOutput), and then ending with the marker for `completed`."
    );

    let response = lm
        .call(
            Chat::new(vec![
                Message::system("You are a helpful assistant."),
                Message::user("Hello, world!"),
            ]),
            "test",
            "[[ ## output ## ]]\n{\"reasoning\": \"The capital of France is Paris.\", \"rating\": 5}\n\n[[ ## completed ## ]]".to_string(),
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response.0);

    assert_eq!(output.len(), 1);

    let parsed_output: serde_json::Value =
        serde_json::from_str("{\"reasoning\": \"The capital of France is Paris.\", \"rating\": 5}")
            .unwrap();
    assert_eq!(
        output.get("output").unwrap()["reasoning"],
        parsed_output["reasoning"]
    );
    assert_eq!(
        output.get("output").unwrap()["rating"],
        parsed_output["rating"]
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_demos() {
    let mut signature = sign! {
        (problem: String) -> answer: String
    };

    let adapter = ChatAdapter;

    // Create demo examples
    let demo1 = Example::new(
        hashmap! {
            "problem".to_string() => "What is 2 + 2?".to_string().into(),
            "answer".to_string() => "4".to_string().into(),
        },
        vec!["problem".to_string()],
        vec!["answer".to_string()],
    );

    let demo2 = Example::new(
        hashmap! {
            "problem".to_string() => "What is the largest planet?".to_string().into(),
            "answer".to_string() => "Jupiter".to_string().into(),
        },
        vec!["problem".to_string()],
        vec!["answer".to_string()],
    );

    signature.set_demos(vec![demo1, demo2]).unwrap();

    let current_input = Example::new(
        hashmap! {
            "problem".to_string() => "What is the capital of France?".to_string().into(),
        },
        vec!["problem".to_string()],
        vec!["answer".to_string()],
    );

    let messages: Chat = adapter.format(&signature, current_input);

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    // Should have system message + 2 demo pairs (user + assistant) + current user message
    assert_eq!(messages.len(), 6);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");
    assert_eq!(json[2]["role"], "assistant");
    assert_eq!(json[3]["role"], "user");
    assert_eq!(json[4]["role"], "assistant");
    assert_eq!(json[5]["role"], "user");

    // Check demo 1 formatting
    assert!(
        json[1]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## problem ## ]]\nWhat is 2 + 2?")
    );
    assert!(
        json[2]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## answer ## ]]\n4")
    );
    assert!(
        json[2]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## completed ## ]]")
    );

    // Check demo 2 formatting
    assert!(
        json[3]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## problem ## ]]\nWhat is the largest planet?")
    );
    assert!(
        json[4]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## answer ## ]]\nJupiter")
    );
    assert!(
        json[4]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## completed ## ]]")
    );

    // Check current input formatting
    assert!(
        json[5]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## problem ## ]]\nWhat is the capital of France?")
    );
    assert!(
        json[5]["content"]
            .as_str()
            .unwrap()
            .contains("Respond with the corresponding output fields")
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_empty_demos() {
    let mut signature = sign! {
        (problem: String) -> answer: String
    };

    let adapter = ChatAdapter;

    let current_input = Example::new(
        hashmap! {
            "problem".to_string() => "What is the capital of France?".to_string().into(),
        },
        vec!["problem".to_string()],
        vec!["answer".to_string()],
    );
    signature.set_demos(vec![]).unwrap();

    let messages: Chat = adapter.format(&signature, current_input);

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    // Should only have system message + current user message (no demos)
    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    // Check current input formatting
    assert!(
        json[1]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## problem ## ]]\nWhat is the capital of France?")
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_demo_format_multiple_fields() {
    let mut signature = TestSignature::new();

    let adapter = ChatAdapter;

    let demo = Example::new(
        hashmap! {
            "problem".to_string() => "What is 5 * 6?".to_string().into(),
            "hint".to_string() => "Think about multiplication".to_string().into(),
            "reasoning".to_string() => "5 multiplied by 6 equals 30".to_string().into(),
            "answer".to_string() => "30".to_string().into(),
        },
        vec!["problem".to_string(), "hint".to_string()],
        vec!["reasoning".to_string(), "answer".to_string()],
    );

    signature.set_demos(vec![demo]).unwrap();

    let current_input = Example::new(
        hashmap! {
            "problem".to_string() => "What is 3 + 7?".to_string().into(),
            "hint".to_string() => "Simple addition".to_string().into(),
        },
        vec!["problem".to_string(), "hint".to_string()],
        vec!["reasoning".to_string(), "answer".to_string()],
    );

    let messages: Chat = adapter.format(&signature, current_input);

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    // Should have system + demo user + demo assistant + current user
    assert_eq!(messages.len(), 4);

    // Check demo user message contains both input fields
    assert!(
        json[1]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## problem ## ]]\nWhat is 5 * 6?")
    );
    assert!(
        json[1]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## hint ## ]]\nThink about multiplication")
    );

    // Check demo assistant message contains both output fields and completion marker
    assert!(
        json[2]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## reasoning ## ]]\n5 multiplied by 6 equals 30")
    );
    assert!(
        json[2]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## answer ## ]]\n30")
    );
    assert!(
        json[2]["content"]
            .as_str()
            .unwrap()
            .contains("[[ ## completed ## ]]")
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_cache_hit() {
    let signature = sign! {
        (question: String) -> answer: String
    };

    // Create LM with cache enabled
    let mut config = LMConfig::default();
    config.cache = true;
    
    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();
    
    // Setup the LM client and cache
    lm.setup_client().await;
    
    let adapter = ChatAdapter;
    let lm = Arc::new(Mutex::new(lm));
    
    // Create test input
    let input = Example::new(
        hashmap! {
            "question".to_string() => "What is 2 + 2?".to_string().into(),
        },
        vec!["question".to_string()],
        vec!["answer".to_string()],
    );
    
    // Mock the first response by inserting directly into cache
    let mut output_data = std::collections::HashMap::new();
    output_data.insert("answer".to_string(), serde_json::json!("4"));
    let cached_prediction = Prediction::new(output_data, dspy_rs::LmUsage::default());
    
    // Insert into cache
    {
        let lm_guard = lm.lock().await;
        if let Some(cache) = lm_guard.cache_handler.as_ref() {
            cache.insert(input.clone(), cached_prediction.clone()).unwrap();
        }
    }
    
    // Call adapter - should hit cache
    let result = adapter.call(lm.clone(), &signature, input.clone()).await.unwrap();
    
    assert_eq!(result.data.get("answer").unwrap(), &serde_json::json!("4"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_cache_miss_different_inputs() {
    // Create LM with cache enabled
    let mut config = LMConfig::default();
    config.cache = true;
    
    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();
    
    // Setup the LM client and cache
    lm.setup_client().await;
    
    let lm = Arc::new(Mutex::new(lm));
    
    // First input
    let input1 = Example::new(
        hashmap! {
            "question".to_string() => "What is 2 + 2?".to_string().into(),
        },
        vec!["question".to_string()],
        vec!["answer".to_string()],
    );
    
    // Cache first input
    let mut output1 = std::collections::HashMap::new();
    output1.insert("answer".to_string(), serde_json::json!("4"));
    let prediction1 = Prediction::new(output1, dspy_rs::LmUsage::default());
    
    {
        let lm_guard = lm.lock().await;
        if let Some(cache) = lm_guard.cache_handler.as_ref() {
            cache.insert(input1.clone(), prediction1.clone()).unwrap();
        }
    }
    
    // Second (different) input
    let input2 = Example::new(
        hashmap! {
            "question".to_string() => "What is 3 + 3?".to_string().into(),
        },
        vec!["question".to_string()],
        vec!["answer".to_string()],
    );
    
    // Check that second input is not cached
    {
        let lm_guard = lm.lock().await;
        if let Some(cache) = lm_guard.cache_handler.as_ref() {
            let cached = cache.get(input2.clone()).await.unwrap();
            assert!(cached.is_none());
        }
    }
    
    // But first input should still be cached
    {
        let lm_guard = lm.lock().await;
        if let Some(cache) = lm_guard.cache_handler.as_ref() {
            let cached = cache.get(input1.clone()).await.unwrap();
            assert!(cached.is_some());
            assert_eq!(cached.unwrap().data.get("answer").unwrap(), &serde_json::json!("4"));
        }
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_cache_disabled() {
    // Create LM with cache disabled
    let mut config = LMConfig::default();
    config.cache = false;
    
    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();
    
    // Setup the LM client (no cache will be initialized)
    lm.setup_client().await;
    
    let lm = Arc::new(Mutex::new(lm));
    
    // Verify cache handler is None when cache is disabled
    {
        let lm_guard = lm.lock().await;
        assert!(lm_guard.cache_handler.is_none());
    }
}
