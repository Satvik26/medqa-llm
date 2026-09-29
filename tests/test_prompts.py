from medqa.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_user_message,
    clean_generation,
    render_example,
    render_prompt,
)


def test_closed_book_message_is_just_the_question():
    assert build_user_message("  What is X? ") == "What is X?"


def test_contexts_are_numbered():
    msg = build_user_message("Q?", ["first", "second"])
    assert "[1] first" in msg and "[2] second" in msg and msg.endswith("Question: Q?")


def test_messages_roles():
    roles = [m["role"] for m in build_messages("Q?", answer="A.")]
    assert roles == ["system", "user", "assistant"]


def test_prompt_and_example_share_prefix(tokenizer):
    prompt = render_prompt(tokenizer, "Q?")
    example = render_example(tokenizer, "Q?", "A.")
    assert SYSTEM_PROMPT in prompt
    assert prompt.endswith("<assistant>")
    assert example.startswith(prompt.removesuffix("<assistant>"))
    assert "A." in example


def test_clean_generation_strips_thinking_and_stop_tokens():
    raw = "<think>internal reasoning</think>\n Insulin lowers glucose.<|im_end|>junk"
    assert clean_generation(raw, stop_tokens=["<|im_end|>"]) == "Insulin lowers glucose."
    assert clean_generation("<think>never closed") == ""
