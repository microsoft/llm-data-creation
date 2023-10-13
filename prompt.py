import json
from textwrap import dedent
from typing import List, Optional


def system_instruction(num_examples: int = 3, prompt_type: str = "fix") -> str:
    if prompt_type == "fix":
        instruction = dedent(
            f"""
            - You are creating {num_examples} more examples that follow the format of the example provided, but with a different content.
            - The created examples **must** all have different answers.
            - The created examples **must** have the same options as the provided example.
            - The output **must** be in unnumbered JSON format.
            """
        )
    else:
        instruction = dedent(
            f"""
            - You are creating {num_examples} more examples that follow the format of the example provided, but with a different content.
            - The created examples **must** all have different answers.
            - The output **must** be in unnumbered JSON format.
            """
        )
    return instruction


def example_instruction(
    label_space: List[str],
    question: str,
    answer: str,
    context: Optional[str],
    prompt_type: str = "fix",
) -> str:
    if prompt_type == "fix":
        if context:
            prompt = dedent(
                f"""
                "Options": {json.dumps(label_space)},
                "Answer": "{answer}",
                "Question": "{question}",
                "Context": "{context}"
                """
            )
        else:
            prompt = dedent(
                f"""
                "Options": {json.dumps(label_space)},
                "Answer": "{answer}",
                "Question": "{question}"
                """
            )
    elif prompt_type == "variant":
        if context:
            prompt = dedent(
                f"""
                "Question": "{question}",
                "Context": "{context}",
                "Options": {json.dumps(label_space)},
                "Answer": "{answer}"
                """
            )
        else:
            prompt = dedent(
                f"""
                "Question": "{question}",
                "Options": {json.dumps(label_space)},
                "Answer": "{answer}"
                """
            )

    return "{" + prompt + "}"
