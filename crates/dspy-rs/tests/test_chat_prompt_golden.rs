use dspy_rs::{ChatAdapter, Example, Signature};

#[derive(Signature, Clone, Debug)]
struct GoldenSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[test]
fn golden_system_prompt_is_stable() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<GoldenSig>()
        .expect("system prompt should format");

    let expected = concat!(
        "Your input fields are:\n",
        "1. `question` (string)\n",
        "\n",
        "Your output fields are:\n",
        "1. `answer` (string)\n",
        "\n",
        "All interactions will be structured in the following way, with the appropriate values filled in.\n",
        "\n",
        "[[ ## question ## ]]\n",
        "question\n",
        "\n",
        "[[ ## answer ## ]]\n",
        "Output field `answer` should be of type: string\n",
        "\n",
        "[[ ## completed ## ]]\n",
        "\n",
        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.\n",
        "\n",
        "In adhering to this structure, your objective is: \n",
        "        Given the fields `question`, produce the fields `answer`.",
    );

    assert_eq!(system, expected);
}

#[test]
fn golden_user_prompt_is_stable() {
    let adapter = ChatAdapter;
    let input = GoldenSigInput {
        question: "What is 2+2?".to_string(),
    };
    let user = adapter.format_user_message_typed::<GoldenSig>(&input);

    let expected = concat!(
        "[[ ## question ## ]]\n",
        "What is 2+2?\n",
        "\n",
        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    );

    assert_eq!(user, expected);
}

#[test]
fn golden_assistant_prompt_is_stable() {
    let adapter = ChatAdapter;
    let output = GoldenSigOutput {
        answer: "4".to_string(),
    };
    let assistant = adapter.format_assistant_message_typed::<GoldenSig>(&output);

    let expected = concat!(
        "[[ ## answer ## ]]\n",
        "4\n",
        "\n",
        "[[ ## completed ## ]]\n",
    );
    assert_eq!(assistant, expected);
}

#[test]
fn golden_demo_messages_are_stable() {
    let adapter = ChatAdapter;
    let demo = Example::<GoldenSig>::new(
        GoldenSigInput {
            question: "What is 2+2?".to_string(),
        },
        GoldenSigOutput {
            answer: "4".to_string(),
        },
    );

    let (user, assistant) = adapter.format_demo_typed::<GoldenSig>(&demo);

    let expected_user = concat!(
        "[[ ## question ## ]]\n",
        "What is 2+2?\n",
        "\n",
        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    );
    let expected_assistant = concat!(
        "[[ ## answer ## ]]\n",
        "4\n",
        "\n",
        "[[ ## completed ## ]]\n",
    );

    assert_eq!(user, expected_user);
    assert_eq!(assistant, expected_assistant);
}
