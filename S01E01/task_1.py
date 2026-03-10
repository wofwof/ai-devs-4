import csv
import json
from os import getenv
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from common.hub_client import HubClient
from job_description_schema import JOB_DESCRIPTION_PROMPT, JOB_DESCRIPTION_SCHEMA

load_dotenv()

DATA_FILE = Path(__file__).parent / "people.csv"
CURRENT_YEAR = 2026

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=getenv("OPENROUTER_API_KEY")
)


def load_people():
    with open(DATA_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_candidates(people: list[dict]) -> list[dict]:
    results = []
    for person in people:
        if person["gender"] != "M":
            continue
        if person["birthPlace"] != "Grudziądz":
            continue
        birth_year = int(person["birthDate"].split("-")[0])
        age = CURRENT_YEAR - birth_year
        if 20 <= age <= 40:
            results.append({**person, "age": age})
    return results


people = load_people()
candidates = filter_candidates(people)

# Build user content with candidate data for tagging
candidates_data = "\n".join(
    f"[{i}] {c['name']} {c['surname']}: {c['job']}" for i, c in enumerate(candidates)
)

print(candidates_data)

completion = client.chat.completions.create(
    model="openai/gpt-5.2",
    response_format=JOB_DESCRIPTION_SCHEMA,
    messages=[
        {"role": "system", "content": JOB_DESCRIPTION_PROMPT},
        {"role": "user", "content": candidates_data},
    ],
)
print(completion)

tagged = json.loads(completion.choices[0].message.content)["people"]

# Filter only people with "transport" tag and format for API
transport_candidates = []
for entry in tagged:
    if "transport" in entry["tags"]:
        c = candidates[entry["index"]]
        transport_candidates.append(
            {
                "name": c["name"],
                "surname": c["surname"],
                "gender": c["gender"],
                "born": int(c["birthDate"].split("-")[0]),
                "city": c["birthPlace"],
                "tags": entry["tags"],
            }
        )

with open(Path(__file__).parent / "transport_candidates.txt", "w", encoding="utf-8") as f:
    f.write(f"Candidates with 'transport' tag ({len(transport_candidates)}):\n")
    for c in transport_candidates:
        f.write(f"  {c['name']} {c['surname']}, born {c['born']}, tags: {c['tags']}\n")

hub = HubClient()
result = hub.verify(task="people", answer=transport_candidates)
print(result)
