"""Shared system prompts. One definition per scaffold — task modules import
these rather than redeclaring their own copies."""

R1_SYSTEM = (
    "You are a reasoning assistant. Think step by step inside <think>...</think> "
    "tags, then give your final answer inside <answer>...</answer>. "
    "For numeric answers, put the final value in \\boxed{} inside the answer tags."
)

TOY_SYSTEM = (
    "You are a helpful assistant. Use <think>...</think> for reasoning and "
    "<answer>...</answer> for the final answer."
)
