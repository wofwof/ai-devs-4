"""S01E04 — orchestrator with image recognizer agent."""

import asyncio
from common.tools import registry, FETCH_URL
from common.agents import AgentDef, AgentRegistry
from common.chat import chat
from common.repl import run_repl
from common.mcp_client import mcp_tools


async def main():
    async with mcp_tools(registry) as mcp_tool_names:
        agents = AgentRegistry(registry)

        agents.register(AgentDef(
            name="image_analyzer",
            description="Analyzes images and returns detailed descriptions.",
            system_prompt="You are an image analysis specialist. The image is provided directly in the message. Describe what you see in detail. Answer in the same language as the user's request.",
            tools=[FETCH_URL, *mcp_tool_names],
            model="google/gemini-3-flash-preview",
            temperature=0.2,
            max_steps=5,
            meta_schema={
                "imagePath": {
                    "type": "string",
                    "description": "URL or local path to the image to analyze",
                },
            },
        ))

        await run_repl(
            chat=chat,
            registry=registry,
            model="openai/gpt-4.1",
            system_prompt=(
                "You are an autonomous orchestrator.\n"
                "RULES:\n"
                "- NEVER stop to wait for user input mid-task. Complete the full task in one go using tool calls.\n"
                "- NEVER ask the user to confirm, clarify, or provide missing info. Figure it out yourself from available data.\n"
                "- You do NOT know the current date. ALWAYS call today_date tool when you need it. NEVER guess.\n"
                "- You CANNOT analyze images yourself. When you need information from an image (PNG, JPG), delegate to the image_analyzer agent with imagePath.\n"
                "- All local files are stored in ./workspace. fetch_url saves there by default. Use MCP fs_read with relative paths (e.g. 'dane.txt') to read them.\n"
                "- When reading files via MCP, strip line number prefixes (e.g. '1|') — those are display artifacts.\n"
                "- When documentation references other files ([include file=\"...\"]), fetch them too — they contain critical details.\n"
                "- Cross-reference ALL available data before filling in any field. If data exists across multiple documents or images, combine it.\n"
                "- For long/structured answers: save to a file via fs_write, then pass the filename to send_solution.\n"
                "- The task name for send_solution comes from the user. NEVER guess it.\n"
            ),
        )


asyncio.run(main())
