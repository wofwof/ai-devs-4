from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, unquote

import requests


@dataclass
class ToolResult:
    """Result returned by a tool handler.

    data: ready-to-use content (goes straight to agent), or a prompt for a sub-LLM call.
    llm_schema: if set, the main loop makes an LLM call with `data` as prompt
                and this as response_format, then feeds the LLM output to the agent.
    stop: if True, the main loop should stop after this tool call.
    """

    data: str
    llm_schema: dict | None = None
    stop: bool = False

    @property
    def needs_llm(self) -> bool:
        return self.llm_schema is not None


@dataclass
class _Tool:
    """Internal representation of a registered tool."""

    schema: dict
    handler: Callable[..., ToolResult]


class ToolRegistry:
    """Central registry for OpenAI function-calling tools and their handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, _Tool] = {}

    def register(
        self,
        schema: dict,
        handler: Callable[..., ToolResult],
    ) -> None:
        """Register a tool with its OpenAI schema and handler function.

        The tool name is extracted from the schema automatically.
        """
        name = schema["function"]["name"]
        self._tools[name] = _Tool(schema=schema, handler=handler)

    def get_tools(self, *names: str) -> list[dict]:
        """Return OpenAI tool schemas for the given tool names.

        If no names are provided, returns all registered tool schemas.
        """
        if not names:
            return [t.schema for t in self._tools.values()]
        missing = set(names) - self._tools.keys()
        if missing:
            raise KeyError(f"Unknown tools: {missing}")
        return [self._tools[n].schema for n in names]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool handler by name with the given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(data=f"Error: unknown tool '{name}'")
        return tool.handler(**arguments)

    @property
    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())


# Global registry instance
registry = ToolRegistry()


# --- Built-in tools ---

FETCH_URL = "fetch_url"

FETCH_URL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch content from a URL. For text responses (HTML, JSON, plain text) returns the body as text. For binary files (images, PDFs, archives, etc.) downloads them to a local path and returns the file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch or download from",
                },
                "save_dir": {
                    "type": "string",
                    "description": "Directory to save downloaded binary files. Defaults to ./workspace.",
                },
                "filename": {
                    "type": "string",
                    "description": "Override the filename for saved files. If not provided, derived from the URL or Content-Disposition header.",
                },
            },
            "required": ["url"],
        },
    },
}

TEXT_CONTENT_TYPES = {"text/", "application/json", "application/xml", "application/xhtml"}


def _is_text_response(content_type: str) -> bool:
    return any(ct in content_type for ct in TEXT_CONTENT_TYPES)


def _extract_filename(response: requests.Response, url: str) -> str:
    cd = response.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        parts = cd.split("filename=")[-1].strip().strip('"').strip("'")
        if parts:
            return parts

    path = urlparse(url).path
    name = unquote(path.split("/")[-1])
    return name if name else "downloaded_file"


def handle_fetch_url(url: str, save_dir: str = "workspace", filename: str = "") -> ToolResult:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        return ToolResult(data=f"Error fetching {url}: {e}")

    content_type = response.headers.get("Content-Type", "")

    if _is_text_response(content_type):
        return ToolResult(data=response.text)

    name = filename or _extract_filename(response, url)
    save_path = Path(save_dir) / name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(response.content)
    return ToolResult(data=f"Downloaded to {save_path} ({len(response.content)} bytes)")


registry.register(FETCH_URL_SCHEMA, handle_fetch_url)


# --- today_date tool ---

TODAY_DATE = "today_date"

TODAY_DATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "today_date",
        "description": "Returns today's date in YYYY-MM-DD format.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


def handle_today_date() -> ToolResult:
    from datetime import date
    return ToolResult(data=date.today().isoformat())


registry.register(TODAY_DATE_SCHEMA, handle_today_date)


# --- send_solution tool ---

SEND_SOLUTION = "send_solution"

SEND_SOLUTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_solution",
        "description": "Submit the final answer for a course task to the verification hub. Just provide the task name and plain text answer — the handler takes care of JSON formatting. For long answers, save to a file first and pass the filename.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task name for verification (e.g. 'sendit', 'category')",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer as plain text. For long answers, pass a workspace filename instead (e.g. 'declaration.txt').",
                },
            },
            "required": ["task", "answer"],
        },
    },
}


def _detect_task_folder() -> str:
    """Detect SXXEXX folder from sys.argv (e.g. 'S01E04/task.py' -> 'S01E04')."""
    import sys, re
    for arg in sys.argv:
        m = re.search(r'(S\d{2}E\d{2})', arg)
        if m:
            return m.group(1)
    return "unknown"


def _log_answer(task_folder: str, answer_str: str) -> None:
    """Append answer to the answers log file."""
    answers_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "answers.md"
    line = f"{task_folder}: {answer_str}\n"
    with open(answers_path, "a", encoding="utf-8") as f:
        f.write(line)


def handle_send_solution(task: str, answer: str) -> ToolResult:
    from common.hub_client import HubClient

    # If answer looks like a filename, read from workspace
    raw_answer = answer
    if not answer.startswith("{") and not answer.startswith("["):
        candidate = Path("workspace") / answer.removeprefix("workspace/").removeprefix("workspace\\")
        if candidate.is_file():
            raw_answer = candidate.read_text(encoding="utf-8")
            print(f"  READ FROM FILE: {candidate}")

    # Try to parse as JSON first (agent may have sent a JSON object/array)
    try:
        parsed_answer = json.loads(raw_answer)
    except (json.JSONDecodeError, ValueError):
        # Plain text — wrap it so hub always gets valid JSON
        parsed_answer = raw_answer

    answer_str = json.dumps(parsed_answer, ensure_ascii=False)
    print(f"  SUBMIT: task={task}, answer={answer_str[:300]}")
    result = HubClient().verify(task=task, answer=parsed_answer)

    # Log answer to file
    task_folder = _detect_task_folder()
    _log_answer(task_folder, answer_str)

    return ToolResult(data=json.dumps(result, ensure_ascii=False), stop=True)


registry.register(SEND_SOLUTION_SCHEMA, handle_send_solution)


# --- Agent tool schema & handler factory ---

def build_agent_schema(name: str, description: str, meta_schema: dict | None = None) -> dict:
    """Build an OpenAI tool schema for an agent tool.

    Args:
        meta_schema: extra properties the orchestrator must provide (e.g. {"imagePath": {"type": "string"}}).
                     These are added to the tool parameters alongside 'task'.
    """
    properties: dict = {
        "task": {
            "type": "string",
            "description": "The task description to send to this agent.",
        },
    }
    required = ["task"]

    if meta_schema:
        properties.update(meta_schema)
        required.extend(meta_schema.keys())

    return {
        "type": "function",
        "function": {
            "name": f"agent_{name}",
            "description": f"Delegate task to the '{name}' agent: {description}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_agent_handler(agent, runner, tool_registry: ToolRegistry):
    """Create an async handler that runs a sub-agent and returns a ToolResult.

    Args:
        agent: AgentDef instance
        runner: async (agent, task, registry, meta) -> str
        tool_registry: ToolRegistry the agent can use
    """
    async def handler(task: str, **meta) -> ToolResult:
        result = await runner(agent, task, tool_registry, meta)
        return ToolResult(data=result)
    return handler
