"""Agent-as-Tool pattern: register sub-agents that the orchestrator LLM can invoke like tools."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from common.ai_client import ai_client
from common.tools import ToolRegistry, ToolResult, build_agent_schema, build_agent_handler


@dataclass
class AgentDef:
    """Definition of a sub-agent."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = "openai/gpt-4.1-mini"
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int = 10
    response_format: dict | None = None
    top_p: float | None = None
    meta_schema: dict | None = None


def _resolve_image_url(image_path: str) -> str | None:
    """Convert an image path to a URL suitable for multimodal LLM content.

    - HTTP(S) URLs are returned as-is.
    - Local file paths are read and converted to base64 data URIs.
    """
    if image_path.startswith(("http://", "https://")):
        return image_path

    path = Path(image_path)
    if not path.exists():
        print(f"    [image] WARNING: file not found: {path}")
        return None

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    print(f"    [image] loaded {path} ({len(data)} bytes base64, {mime_type})")
    return f"data:{mime_type};base64,{data}"


async def run_agent(
    agent: AgentDef,
    task: str,
    tool_registry: ToolRegistry,
    meta: dict | None = None,
) -> str:
    """Run a sub-agent loop to completion, return final text."""
    import asyncio

    user_content_parts: list[dict] = [{"type": "text", "text": task}]

    if meta:
        image_path = meta.pop("imagePath", None)
        if image_path:
            image_url = _resolve_image_url(image_path)
            if image_url:
                user_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })
        if meta:
            user_content_parts[0]["text"] += f"\n\nMetadata:\n{json.dumps(meta, ensure_ascii=False)}"

    history: list[dict] = [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": user_content_parts},
    ]

    prefix = f"    [{agent.name}]"

    for step in range(1, agent.max_steps + 1):
        print(f"{prefix} step {step}/{agent.max_steps} — calling LLM ({agent.model})")

        call_kwargs: dict = {
            "model": agent.model,
            "messages": history,
        }
        if agent.temperature is not None:
            call_kwargs["temperature"] = agent.temperature
        if agent.max_tokens is not None:
            call_kwargs["max_tokens"] = agent.max_tokens
        if agent.top_p is not None:
            call_kwargs["top_p"] = agent.top_p
        if agent.response_format is not None:
            call_kwargs["response_format"] = agent.response_format

        agent_tools = tool_registry.get_tools(*agent.tools) if agent.tools else []
        if agent_tools:
            call_kwargs["tools"] = agent_tools

        response = ai_client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message

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
            print(f"{prefix} step {step} — LLM responded (no tool calls), finishing")
            return message.content or ""

        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"{prefix} TOOL CALL: {name}({json.dumps(args, ensure_ascii=False)[:200]})")

            tool_result = tool_registry.execute(name, args)
            if asyncio.iscoroutine(tool_result):
                tool_result = await tool_result

            if tool_result.needs_llm:
                print(f"{prefix} LLM SUB-CALL: processing tool output with LLM...")
                sub = ai_client.chat.completions.create(
                    model=agent.model,
                    response_format=tool_result.llm_schema,
                    messages=[{"role": "user", "content": tool_result.data}],
                )
                tool_result.data = sub.choices[0].message.content

            print(f"{prefix} TOOL RESULT: {tool_result.data[:200]}")

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result.data,
            })

            if tool_result.stop:
                print(f"{prefix} STOP: tool '{name}' requested stop")
                return tool_result.data

    print(f"{prefix} max steps ({agent.max_steps}) reached, returning last response")
    return history[-1].get("content", "") if history else ""


class AgentRegistry:
    """Registers AgentDefs as tools in a ToolRegistry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._agents: dict[str, AgentDef] = {}

    def register(self, agent: AgentDef) -> None:
        """Register an agent — creates a corresponding tool in the ToolRegistry."""
        self._agents[agent.name] = agent
        schema = build_agent_schema(agent.name, agent.description, agent.meta_schema)
        handler = build_agent_handler(agent, run_agent, self._tool_registry)
        self._tool_registry.register(schema, handler)
