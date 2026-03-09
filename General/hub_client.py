import requests
from os import getenv


class HubClient:
    URL = "https://hub.ag3nts.org/verify"

    def __init__(self):
        self.api_key = getenv("AI_DEVS_KEY")

    def verify(self, task: str, answer) -> dict:
        payload = {
            "apikey": self.api_key,
            "task": task,
            "answer": answer,
        }
        response = requests.post(self.URL, json=payload)
        response.raise_for_status()
        return response.json()
