"""Chat function with tool-calling support for the REPL."""

from __future__ import annotations

import asyncio
import json
from common.ai_client import ai_client
from common.tools import ToolRegistry


async def chat(
    user_input: str,
    history: list[dict],
    *,
    registry: ToolRegistry | None = None,
    model: str = "openai/gpt-5.2-pro",  # "openai/gpt-4.1-mini",
    **kwargs,
) -> dict:
    """Send user input to LLM, handle tool calls, return response + updated history.

    Args:
        registry: ToolRegistry to use for tool execution. If None, no tools.
        model: LLM model to use.
        **kwargs: extra kwargs forwarded to the LLM call (temperature, etc.)
    """
    history.append({"role": "user", "content": user_input})

    tools = registry.get_tools() if registry else []

    while True:
        call_kwargs = {
            "model": model,
            "messages": history,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools

        response = ai_client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message

        # Append assistant message to history
        assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        history.append(assistant_msg)

        if not message.tool_calls:
            return {
                "response": message.content or "",
                "conversation_history": history,
            }

        # Execute tool calls
        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            is_agent = name.startswith("agent_")
            if is_agent:
                agent_name = name.removeprefix("agent_")
                task_preview = args.get("task", "")[:100]
                meta = {k: v for k, v in args.items() if k != "task"}
                print(f"\n{'='*60}")
                print(f"  AGENT SPAWN: {agent_name}")
                print(f"  task: {task_preview}")
                if meta:
                    print(f"  meta: {json.dumps(meta, ensure_ascii=False)}")
                print(f"{'='*60}")
            else:
                print(
                    f"  TOOL CALL: {name}({json.dumps(args, ensure_ascii=False)[:200]})"
                )

            tool_result = registry.execute(name, args)
            if asyncio.iscoroutine(tool_result):
                tool_result = await tool_result

            if tool_result.needs_llm:
                print(f"  LLM SUB-CALL: processing tool output with LLM...")
                sub = ai_client.chat.completions.create(
                    model=call_kwargs["model"],
                    response_format=tool_result.llm_schema,
                    messages=[{"role": "user", "content": tool_result.data}],
                )
                tool_result.data = sub.choices[0].message.content

            if is_agent:
                print(f"{'='*60}")
                print(f"  AGENT DONE: {agent_name}")
                print(f"  result: {tool_result.data[:300]}")
                print(f"{'='*60}\n")
            else:
                print(f"  TOOL RESULT: {tool_result.data[:200]}")

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result.data,
                }
            )

            if tool_result.stop:
                print(f"  STOP: tool '{name}' requested stop")
                return {
                    "response": tool_result.data,
                    "conversation_history": history,
                }
