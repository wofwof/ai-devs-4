import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools import TOOLS, execute_tool_call

load_dotenv()

ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """Znajdź agenta tajniaka — osobę, która była najbliżej dowolnej elektrowni.

Wykonaj kroki DOKŁADNIE w tej kolejności:

1. Użyj read_user_info — pobierz listę kandydatów (imię, nazwisko, rok urodzenia).
2. Użyj read_plants_data — pobierz dane elektrowni (miasta i kody).
3. Użyj get_plant_coordinates — podaj nazwy miast elektrowni, dostaniesz ich koordynaty.
4. Dla KAŻDEGO kandydata użyj get_person_locations — pobierz historię lokalizacji.
5. Dla KAŻDEGO kandydata użyj find_nearest_plant — podaj jego lokalizacje i koordynaty WSZYSTKICH elektrowni. Zapisz wynik (min_distance_km i nearest_plant_city).
6. Porównaj min_distance_km wszystkich kandydatów. Wybierz tego z NAJMNIEJSZĄ odległością — to jest tajniak.
7. TYLKO dla wybranego tajniaka użyj get_access_level (podaj imię, nazwisko, rok urodzenia).
8. Użyj submit_answer — podaj imię, nazwisko, access_level i kod elektrowni (z plants_data dla nearest_plant_city).
9. Jeśli submit_answer zwrócił sukces, użyj save_agent. Jeśli nie — zakończ.

WAŻNE: Nie odpytuj get_access_level dla wszystkich kandydatów — tylko dla jednego wybranego tajniaka."""


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    MAX_STEPS = 15
    step = 0

    while step < MAX_STEPS:
        step += 1
        print(f"\n--- Step {step}/{MAX_STEPS} ---")

        completion = ai_client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=messages,
            tools=TOOLS,
        )

        choice = completion.choices[0]
        message = choice.message

        # Append the full assistant message (may contain tool_calls)
        messages.append(message)

        # If model wants to call tools, execute them and feed results back
        if message.tool_calls:
            should_stop = False
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  Tool call: {name}({args})")

                tool_result = execute_tool_call(name, args)

                if tool_result.needs_llm:
                    sub_completion = ai_client.chat.completions.create(
                        model="openai/gpt-4.1-mini",
                        response_format=tool_result.llm_schema,
                        messages=[{"role": "user", "content": tool_result.data}],
                    )
                    tool_result.data = sub_completion.choices[0].message.content

                print(f"  Result: {tool_result.data[:200]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result.data,
                })

                if tool_result.stop:
                    should_stop = True

            if should_stop:
                print("\nAgent stopped by tool.")
                break
            continue

        # No tool calls — model responded with text
        print(f"Agent: {message.content}")

        if choice.finish_reason == "stop":
            print("\nAgent finished.")
            break
    else:
        print("\nMax steps reached.")
