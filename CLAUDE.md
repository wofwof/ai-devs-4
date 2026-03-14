# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI programming learning project (AI_Devs 4 course). Tasks are completed using LLM APIs via OpenRouter as the gateway. All code is Python. The course hub at `hub.ag3nts.org` is used for task verification.

## Running Code

```bash
source .venv/Scripts/activate   # Windows/Git Bash
python S01E01/task_1.py         # Run any task script
```

To install after cloning:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
```

## Environment Variables (`.env` in project root)

- `OPENROUTER_API_KEY` — LLM API access via OpenRouter
- `AI_DEVS_KEY` — course hub API key for task verification
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` — observability (Langfuse wraps the OpenAI client)
- `SERVER_URL` — ngrok URL for tasks that require a publicly accessible server

## Architecture

**`common/` package** (installed via `pip install -e .`):
- `ai_client.py` — pre-configured OpenAI client pointing at OpenRouter, wrapped with Langfuse for tracing. Import as `from common.ai_client import ai_client`.
- `hub_client.py` — `HubClient` class for submitting answers to `hub.ag3nts.org/verify`. Usage: `HubClient().verify(task="task_name", answer=data)`.
- `tools.py` — `ToolResult` dataclass and `ToolRegistry` for OpenAI function-calling tool loops. `ToolResult.llm_schema` triggers a sub-LLM call; `ToolResult.stop` ends the agent loop.

**Task folders** (`S01E01/`, `S01E02/`, etc.):
- Each is self-contained. Scripts import from `common/` and may define local helpers (schemas, tool definitions).
- Pattern: build messages → call LLM → process response → submit via `HubClient().verify()`.
- Some tasks run agentic tool-calling loops (see `S01E02/task.py`), others are single-shot (see `S01E01/task_1.py`).
- `S01E03/server.py` shows the HTTP server pattern for tasks requiring a webhook/proxy endpoint.

**`mcp/`** — MCP (Model Context Protocol) servers (TypeScript). `mcp_client.py` in project root provides a Python async client for connecting to them.

## Conventions

- Use `.env` for all secrets — never commit it
- Each task script should be self-contained and runnable independently
- Use `from common.ai_client import ai_client` for LLM calls (not raw OpenAI)
- Use `HubClient().verify()` to submit task answers
- Models are specified as OpenRouter paths (e.g., `openai/gpt-4.1-mini`, `openai/gpt-5.2`)
- New reusable tools (schemas + handlers) go in `common/tools.py`, not in task scripts
