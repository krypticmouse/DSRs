use dsrs::premitives::lm::{LM, LMConfig, LMProvider};
use openai_api_rs::v1::chat_completion::{ChatCompletionMessage, Content, MessageRole};

async fn openai_test() {
    let messages = vec![ChatCompletionMessage {
        name: None,
        tool_calls: None,
        tool_call_id: None,
        role: MessageRole::user,
        content: Content::Text("What is the capital of France?".to_string()),
    }];

    let api_key = std::env::var("OPENAI_API_KEY").unwrap();

    let mut lm = LM {
        provider: LMProvider::OpenAI,
        api_key,
        model: "gpt-4o-mini".to_string(),
        base_url: None,
        lm_config: LMConfig {
            temperature: 0.7,
            top_p: 1.0,
            max_tokens: 100,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop: None,
            n: 1,
        },
        history: vec![],
    };

    let output = lm.forward(messages, "QASignature".to_string()).await;
    println!("OpenAI: {output}");
}

async fn anthropic_test() {
    let messages = vec![ChatCompletionMessage {
        name: None,
        tool_calls: None,
        tool_call_id: None,
        role: MessageRole::user,
        content: Content::Text("What is the capital of France?".to_string()),
    }];

    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();

    let mut lm = LM {
        provider: LMProvider::Anthropic,
        api_key,
        model: "claude-3-5-sonnet-20240620".to_string(),
        base_url: None,
        lm_config: LMConfig {
            temperature: 0.7,
            top_p: 1.0,
            max_tokens: 100,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop: None,
            n: 1,
        },
        history: vec![],
    };

    let output = lm.forward(messages, "QASignature".to_string()).await;
    println!("Anthropic: {output}");
}

#[tokio::main]
async fn main() {
    openai_test().await;
    anthropic_test().await;
}
