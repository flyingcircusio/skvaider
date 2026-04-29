#!/usr/bin/env python3
"""One-off script to test tool calling via an OpenAI-compatible API."""

import json
import sys

from openai import OpenAI

BASE_URL = "http://localhost:8100/v1"
API_KEY = ""

# MODEL = "gpt-oss:120b"
MODEL = "gemma"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'Berlin, Germany'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "What's the weather like in Hamburg right now?"}
]

print(f"Calling {MODEL} at {BASE_URL} ...")

with client.chat.completions.stream(
    model=MODEL,
    messages=messages,  # pyright: ignore[reportArgumentType]
    tools=tools,  # pyright: ignore[reportArgumentType]
    tool_choice="auto",
) as stream:
    for event in stream:
        pass
    final = stream.get_final_completion()

choice = final.choices[0]

if not choice.message.tool_calls:
    print("Expected tool call, but got none.")
    sys.exit(1)


errors = 0

for i, call in enumerate(choice.message.tool_calls):
    print()
    print(f"Tool call #{i}")
    print("============")
    print()

    if call.id:
        print(f"[✅] Call ID: {call.id}")
    else:
        errors += 1
        print(f"[❌] Call ID: {call.id!r}")

    if call.function.name is None:  # pyright: ignore[reportUnnecessaryComparison]
        errors += 1
        print(f"[❌] Functiob name: {call.function.name!r}")
    else:
        print(f"[✅] Function name: {call.function.name}")

    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError as e:
        print(f"[❌] Arguments (raw): {call.function.arguments!r}")
        print(f"     Could not decode argument JSON: {e}")
        errors += 1
    else:
        print(f"[✅] Arguments (parsed): {args}")


if errors:
    print()
    print(f"CRITICAL - {errors} errors")
    sys.exit(1)

print()
print("OK")
