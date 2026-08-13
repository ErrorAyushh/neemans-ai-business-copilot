import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("SARVAM AI API TEST")
print("=" * 70)

API_KEY = os.getenv("SARVAM_API_KEY")

if not API_KEY:
    print("\n❌ SARVAM_API_KEY not found")
    print("Add this to your .env:")
    print("SARVAM_API_KEY=your_key_here")
    raise SystemExit(1)

print("\n✅ SARVAM_API_KEY found")
print(f"Key prefix: {API_KEY[:10]}...")

URL = "https://api.sarvam.ai/v1/chat/completions"
MODEL = "sarvam-105b"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a concise business analytics assistant. "
                "Do not provide hidden reasoning or analysis. "
                "Return only the requested answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Reply with exactly this sentence and nothing else: "
                "SARVAM API TEST SUCCESSFUL"
            ),
        },
    ],
    "temperature": 0.1,
    "max_tokens": 500,
}

print("\n📡 Sending request...")
print(f"Endpoint: {URL}")
print(f"Model: {MODEL}")
print(f"max_tokens: {payload['max_tokens']}")

try:
    response = requests.post(
        URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    print("\n" + "=" * 70)
    print("HTTP RESPONSE")
    print("=" * 70)

    print(f"Status code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")

    if response.status_code != 200:
        print("\n❌ API REQUEST FAILED")
        print("\nResponse:")
        print(response.text)
        raise SystemExit(1)

    print("\n📦 RAW RESPONSE")
    print("-" * 70)

    try:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(response.text)
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("RESPONSE ANALYSIS")
    print("=" * 70)

    choices = data.get("choices", [])

    if not choices:
        print("\n❌ No choices returned")
        raise SystemExit(1)

    choice = choices[0]

    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}

    content = message.get("content")
    reasoning = message.get("reasoning_content")

    print(f"\nFinish reason : {finish_reason}")
    print(f"Content       : {repr(content)}")
    print(f"Reasoning     : {repr(reasoning)}")

    usage = data.get("usage", {})

    print("\nUsage:")
    print(json.dumps(usage, indent=2))

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if response.status_code == 200 and content:
        print("\n✅ SARVAM API TEST SUCCESSFUL")
        print(f"\nModel response:\n{content}")

    elif response.status_code == 200 and reasoning and finish_reason == "length":
        print("\n⚠️ API WORKS, BUT COMPLETION WAS TRUNCATED")
        print("\nSarvam returned reasoning_content but no final content.")
        print("The model hit the completion-token limit before producing")
        print("the final answer.")
        print("\nIncrease max_tokens and retry.")

    elif response.status_code == 200:
        print("\n⚠️ API REQUEST SUCCEEDED")
        print("But Sarvam did not return normal message content.")
        print("\nThis means the API key and endpoint are working,")
        print("but the response format/model behavior needs handling.")

    else:
        print("\n❌ API REQUEST FAILED")


except requests.exceptions.Timeout:
    print("\n❌ Request timed out")

except requests.exceptions.ConnectionError as e:
    print("\n❌ Connection error")
    print(str(e))

except requests.exceptions.RequestException as e:
    print("\n❌ Request failed")
    print(str(e))

except Exception as e:
    print("\n❌ Unexpected error")
    print(type(e).__name__)
    print(str(e))