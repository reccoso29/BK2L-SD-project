"""Speech-to-text layer for the baseline interaction pipeline.

This baseline uses typed input so it can run on any laptop without audio
dependencies. A future microphone STT provider can be swapped in by replacing
`get_user_utterance` with an implementation that returns the same dict shape.
"""


def get_user_utterance(prompt: str = "You: ") -> dict:
    """Collect user input and return a normalized STT-style payload.

    Returns:
        {
            "text": str,
            "error": bool,
            "reason": str,
            "source": str
        }
    """
    try:
        raw = input(prompt)
    except EOFError:
        return {"text": "", "error": True, "reason": "eof", "source": "typed"}
    except KeyboardInterrupt:
        return {"text": "", "error": True, "reason": "interrupt", "source": "typed"}

    normalized = raw.strip()
    if not normalized:
        return {"text": "", "error": True, "reason": "empty", "source": "typed"}

    return {
        "text": normalized.lower(),
        "error": False,
        "reason": "ok",
        "source": "typed",
    }


if __name__ == "__main__":
    print(get_user_utterance())
