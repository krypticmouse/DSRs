#![cfg(feature = "rlm")]

use regex::Regex;
use std::sync::LazyLock;

static DSPY_CODE_BLOCK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```(?P<lang>(?i:repl|python))?[ \t]*\r?\n(?P<code>.*?)```")
        .expect("valid DSPy code block regex")
});

static SUBMIT_CALL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\bSUBMIT\s*\(.*?\)").expect("valid SUBMIT call regex")
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Run { code: String, raw: String },
    Submit { code: String, raw: String },
}

impl Command {
    pub fn parse(response: &str) -> Option<Self> {
        let mut blocks = extract_code_blocks(response);
        if let Some(block) = blocks.pop() {
            return Some(command_from_block(block));
        }

        extract_submit_call(response).map(|submit| Command::Submit {
            code: submit.clone(),
            raw: submit,
        })
    }

    pub fn code(&self) -> &str {
        match self {
            Command::Run { code, .. } | Command::Submit { code, .. } => code,
        }
    }

    pub fn raw(&self) -> &str {
        match self {
            Command::Run { raw, .. } | Command::Submit { raw, .. } => raw,
        }
    }

    pub fn is_submit(&self) -> bool {
        matches!(self, Command::Submit { .. })
    }
}

pub fn get_run_command(response: &str) -> Option<Command> {
    Command::parse(response)
}

pub fn get_code_to_run(response: &str) -> Option<String> {
    Command::parse(response).map(|command| command.code().to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CodeBlock {
    raw: String,
    code: String,
    language: Option<String>,
}

fn command_from_block(block: CodeBlock) -> Command {
    if contains_submit_call(&block.code) {
        Command::Submit {
            code: block.code,
            raw: block.raw,
        }
    } else {
        Command::Run {
            code: block.code,
            raw: block.raw,
        }
    }
}

fn extract_code_blocks(response: &str) -> Vec<CodeBlock> {
    DSPY_CODE_BLOCK_PATTERN
        .captures_iter(response)
        .filter_map(|captures| {
            let raw = captures.get(0)?.as_str().to_string();
            let code = captures
                .name("code")
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            let language = captures
                .name("lang")
                .map(|m| m.as_str().to_ascii_lowercase());

            Some(CodeBlock {
                raw,
                code,
                language,
            })
        })
        .collect()
}

fn contains_submit_call(text: &str) -> bool {
    SUBMIT_CALL_PATTERN.is_match(text)
}

fn extract_submit_call(response: &str) -> Option<String> {
    SUBMIT_CALL_PATTERN
        .captures_iter(response)
        .next()
        .and_then(|captures| captures.get(0))
        .map(|m| m.as_str().trim().to_string())
}
