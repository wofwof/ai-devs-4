import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import sys
import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.hub_client import HubClient

load_dotenv()

API_KEY = os.getenv("AI_DEVS_KEY")
LOCATION_URL = "https://hub.ag3nts.org/api/location"
ACCESS_LEVEL_URL = "https://hub.ag3nts.org/api/accesslevel"

# --- File paths ---

PLANTS_FILE = Path(__file__).resolve().parent / "plants.json"
CANDIDATES_FILE = Path(__file__).resolve().parent.parent / "S01E01" / "transport_candidates.txt"

# --- Structured Output schema for coordinates ---

COORDINATES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "coordinates_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "coordinates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["coordinates"],
            "additionalProperties": False,
        },
    },
}

# --- Tool schemas (OpenAI function calling format) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_plants_data",
            "description": "Read the power plants data (JSON with city, code, power, active status).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_user_info",
            "description": "Read the list of transport suspect candidates (name, surname, birth year).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_plant_coordinates",
            "description": "Get GPS coordinates for a list of city names. Returns JSON: {\"coordinates\": [{\"latitude\": float, \"longitude\": float}, ...]} matching the input array by index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of city names, e.g. ['Zabrze', 'Grudziądz', 'Radom']",
                    }
                },
                "required": ["cities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearest_plant",
            "description": "Find the nearest power plant for a person. Takes the person's location history and all plant coordinates. Returns the minimum distance (km), which plant is closest, and which person location was closest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "latitude": {"type": "number"},
                                "longitude": {"type": "number"},
                            },
                            "required": ["latitude", "longitude"],
                        },
                        "description": "Array of GPS points from person's location history.",
                    },
                    "plants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "latitude": {"type": "number"},
                                "longitude": {"type": "number"},
                            },
                            "required": ["city", "latitude", "longitude"],
                        },
                        "description": "Array of power plant locations with city names.",
                    },
                },
                "required": ["person_locations", "plants"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_access_level",
            "description": "Get access level for a person. Requires name, surname and birth year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Person's first name.",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Person's last name.",
                    },
                    "birth_year": {
                        "type": "integer",
                        "description": "Person's birth year.",
                    },
                },
                "required": ["name", "surname", "birth_year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_person_locations",
            "description": "Get location history (list of coordinate strings) for a person by name and surname.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Person's first name.",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Person's last name.",
                    },
                },
                "required": ["name", "surname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer to verify. Call this when you have identified the suspect. This ends the investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Suspect's first name.",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Suspect's last name.",
                    },
                    "access_level": {
                        "type": "integer",
                        "description": "Suspect's access level.",
                    },
                    "power_plant": {
                        "type": "string",
                        "description": "Power plant code, e.g. PWR1234PL.",
                    },
                },
                "required": ["name", "surname", "access_level", "power_plant"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_agent",
            "description": "Save the identified agent data to a file. Call this ONLY after a successful submit_answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Agent's first name.",
                    },
                    "surname": {
                        "type": "string",
                        "description": "Agent's last name.",
                    },
                    "access_level": {
                        "type": "integer",
                        "description": "Agent's access level.",
                    },
                    "power_plant": {
                        "type": "string",
                        "description": "Power plant code.",
                    },
                },
                "required": ["name", "surname", "access_level", "power_plant"],
            },
        },
    },
]

@dataclass
class ToolResult:
    """Result returned by a tool handler.

    data: ready-to-use content (goes straight to agent), or a prompt for a sub-LLM call.
    llm_schema: if set, the main loop makes an LLM call with `data` as prompt
                and this as response_format, then feeds the LLM output to the agent.
    """
    data: str
    llm_schema: dict | None = None
    stop: bool = False

    @property
    def needs_llm(self) -> bool:
        return self.llm_schema is not None


# --- Tool handlers ---


def handle_read_plants_data() -> ToolResult:
    return ToolResult(data=PLANTS_FILE.read_text(encoding="utf-8"))


def handle_read_user_info() -> ToolResult:
    return ToolResult(data=CANDIDATES_FILE.read_text(encoding="utf-8"))


def handle_get_plant_coordinates(cities: list[str]) -> ToolResult:
    prompt = (
        "Return GPS coordinates (latitude, longitude) for each Polish city below. "
        "Keep the same order as the input list.\n\n"
        + json.dumps(cities, ensure_ascii=False)
    )
    return ToolResult(data=prompt, llm_schema=COORDINATES_SCHEMA)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def handle_find_nearest_plant(person_locations: list[dict], plants: list[dict]) -> ToolResult:
    best_distance = float("inf")
    best_plant = None
    best_person_loc = None

    for loc in person_locations:
        for plant in plants:
            dist = _haversine_km(loc["latitude"], loc["longitude"], plant["latitude"], plant["longitude"])
            if dist < best_distance:
                best_distance = dist
                best_plant = plant
                best_person_loc = loc

    result = {
        "min_distance_km": round(best_distance, 2),
        "nearest_plant_city": best_plant["city"],
        "nearest_plant_lat": best_plant["latitude"],
        "nearest_plant_lng": best_plant["longitude"],
        "person_lat": best_person_loc["latitude"],
        "person_lng": best_person_loc["longitude"],
    }
    return ToolResult(data=json.dumps(result))


def handle_get_access_level(name: str, surname: str, birth_year: int) -> ToolResult:
    payload = {"apikey": API_KEY, "name": name, "surname": surname, "birthYear": birth_year}
    response = requests.post(ACCESS_LEVEL_URL, json=payload)
    response.raise_for_status()
    return ToolResult(data=json.dumps(response.json(), ensure_ascii=False))


def handle_get_person_locations(name: str, surname: str) -> ToolResult:
    payload = {"apikey": API_KEY, "name": name, "surname": surname}
    response = requests.post(LOCATION_URL, json=payload)
    response.raise_for_status()
    return ToolResult(data=json.dumps(response.json(), ensure_ascii=False))


def handle_submit_answer(name: str, surname: str, access_level: int, power_plant: str) -> ToolResult:
    answer = {
        "name": name,
        "surname": surname,
        "accessLevel": access_level,
        "powerPlant": power_plant,
    }
    client = HubClient()
    result = client.verify("findhim", answer)
    success = result.get("status_code") == 200
    msg = result.get("message", "No message")
    if success:
        print(f"\n=== SUCCESS ===\n{msg}\n===============")
    else:
        print(f"\n=== FAILED (status {result.get('status_code')}) ===\n{msg}\n===============")
    return ToolResult(data=json.dumps(result), stop=True)


AGENT_FILE = Path(__file__).resolve().parent / "found_agent.json"


def handle_save_agent(name: str, surname: str, access_level: int, power_plant: str) -> ToolResult:
    data = {
        "name": name,
        "surname": surname,
        "accessLevel": access_level,
        "powerPlant": power_plant,
    }
    AGENT_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAgent saved to {AGENT_FILE}")
    return ToolResult(data=f"Agent saved to {AGENT_FILE}", stop=True)


TOOL_HANDLERS = {
    "read_plants_data": handle_read_plants_data,
    "read_user_info": handle_read_user_info,
    "get_plant_coordinates": handle_get_plant_coordinates,
    "find_nearest_plant": handle_find_nearest_plant,
    "get_access_level": handle_get_access_level,
    "get_person_locations": handle_get_person_locations,
    "submit_answer": handle_submit_answer,
    "save_agent": handle_save_agent,
}


def execute_tool_call(name: str, arguments: dict) -> ToolResult:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return ToolResult(data=f"Error: unknown tool '{name}'")
    return handler(**arguments)
