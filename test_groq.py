
import os
import requests
import json
import getpass

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = getpass.getpass("Enter your Groq API key: ")

if not api_key.strip():
    print("ERROR: No API key entered.")
    raise SystemExit

url = "https://api.groq.com/openai/v1/chat/completions"

payload = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly: KnowVia is working."
        }
    ],
    "temperature": 0.1,
    "max_completion_tokens": 300,
    "include_reasoning": False
}

try:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=(10, 90)
    )

    print("\nHTTP STATUS:", response.status_code)

    data = response.json()

    print("\nCOMPLETE RESPONSE:")
    print(json.dumps(data, indent=2))

    if response.status_code == 200:
        choices = data.get("choices", [])

        if choices:
            message = choices[0].get("message", {})
            print("\nEXTRACTED ANSWER:")
            print(repr(message.get("content")))
            print("\nFINISH REASON:")
            print(choices[0].get("finish_reason"))

except Exception as error:
    print("\nTEST FAILED:")
    print(repr(error))
