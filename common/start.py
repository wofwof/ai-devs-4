"""Main entry point — starts the interactive REPL."""

import asyncio
from common.repl import run_repl
from common.chat import chat


if __name__ == "__main__":
    asyncio.run(run_repl(chat=chat))
