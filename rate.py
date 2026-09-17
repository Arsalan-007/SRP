import requests
import json

response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer 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"
  }
)

print(json.dumps(response.json(), indent=2))
