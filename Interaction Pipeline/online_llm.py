"""Cloud LLM placeholder/scaffold for Knightro baseline.

Uses environment variables and does not hardcode secrets.
If configuration is missing or the request fails, returns None so the router
can fall back to offline behavior.
"""

import json
import os
import urllib.error
import urllib.request


SYSTEM_PROMPT = (
    "You are Knightro for UCF interactions. Keep answers short, upbeat, and safe. "
    "Avoid politics or inappropriate content."
)


def _post_openai_compatible_chat(user_text: str) -> str | None:
    """Minimal OpenAI-compatible chat call using stdlib only.

    Required env vars:
      LLM_API_KEY
      LLM_BASE_URL (example: https://api.groq.com/openai/v1)
      LLM_MODEL (example: llama-3.1-8b-instant)
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        print("[online_llm] Cloud config missing (LLM_API_KEY/LLM_BASE_URL/LLM_MODEL).")
        return None

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.3,
        "max_tokens": 120,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
        print(f"[online_llm] Cloud request failed: {exc}")
        return None


def handle_unknown_with_cloud(user_text: str) -> str | None:
    """Return cloud summary for unknown input, or None on failure."""
    answer = _post_openai_compatible_chat(user_text)
    if answer is None:
        return None
    print("[online_llm] Would use cloud LLM response for unknown intent.")
    return f"Would answer using cloud LLM: {answer}"


if __name__ == "__main__":
    print(handle_unknown_with_cloud("Tell me something about UCF engineering."))
