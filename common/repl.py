"""Interactive REPL loop. Delegates AI interaction to a caller-provided chat function."""

from __future__ import annotations

from functools import partial
from typing import Awaitable, Callable

from common.tools import ToolRegistry


ChatFn = Callable[..., Awaitable[dict]]
"""Signature: async (user_input, history, *, registry, model, **kwargs) -> {"response": str, "conversation_history": list}"""


def create_conversation(
    system_prompt: str = "You are a helpful assistant.",
) -> list[dict]:
    """Create a fresh conversation history."""
    return [{"role": "system", "content": system_prompt}]


async def run_repl(
    *,
    chat: ChatFn,
    system_prompt: str = "You are a helpful assistant.",
    registry: ToolRegistry | None = None,
    model: str = "openai/gpt-5.2-pro",
    **kwargs,
) -> None:
    """Run an interactive REPL loop.

    Args:
        chat: async chat function (see common.chat.chat)
        system_prompt: initial system message for the conversation
        registry: ToolRegistry passed to chat on each call
        model: LLM model passed to chat
        **kwargs: extra kwargs forwarded to chat (temperature, etc.)
    """
    history = create_conversation(system_prompt)

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "clear":
            history = create_conversation(system_prompt)
            print("Conversation cleared\n")
            continue

        if not user_input.strip():
            continue

        try:
            result = await chat(user_input, history, registry=registry, model=model, **kwargs)
            history = result["conversation_history"]
            print(f"\nAssistant: {result['response']}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main(
    chat: ChatFn,
    system_prompt: str = "You are a helpful assistant.",
    registry: ToolRegistry | None = None,
    model: str = "openai/gpt-4.1-mini",
    **kwargs,
) -> None:
    """Entry point: run the REPL until the user exits."""
    import asyncio
    asyncio.run(run_repl(
        chat=chat,
        system_prompt=system_prompt,
        registry=registry,
        model=model,
        **kwargs,
    ))
