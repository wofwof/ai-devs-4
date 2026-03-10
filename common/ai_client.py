from os import getenv
from dotenv import load_dotenv
from langfuse.openai import OpenAI

load_dotenv()

ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)
