"""Keyword-based intent detection for Knightro baseline interactions.

The implementation is intentionally simple and isolated so it can be replaced
later with a model-based classifier.
"""

INTENT_KEYWORDS = {
    "identity": [
        "who are you",
        "what are you",
        "introduce yourself",
        "your name",
        "who is knightro",
    ],
    "chant": [
        "ucf chant",
        "chant",
        "sing",
        "cheer",
    ],
    "dance": [
        "dance",
        "show me a dance",
        "do a dance",
        "zombie nation",
    ],
    "goknights": [
        "go knights",
        "go knight",
        "charge on",
        "ucf spirit",
    ],
    "farewell": [
        "goodbye",
        "bye",
        "see you",
        "later",
        "catch you later",
    ],
    "known_person": [
        "president cartwright",
        "john c hitt",
        "terry mohawk",
        "recognized person",
        "known person",
    ],
    "ucf_trivia": [
        "ucf trivia",
        "trivia",
        "when was ucf founded",
        "what is ucf mascot",
        "what are ucf colors",
    ],
    "directions": [
        "where is",
        "how do i get to",
        "directions",
        "engineering atrium",
        "student union",
        "library",
    ],
    "knightro_info": [
        "what is knightro",
        "what can you do",
        "about this project",
        "about knightro",
    ],
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
    ],
}


def detect_intent(text: str) -> str:
    """Return one of: greeting, known_person, ucf_trivia, directions,
    knightro_info, identity, chant, dance, goknights, farewell, unknown.
    """
    normalized = (text or "").strip().lower()
    print(f"[intent_detection] Analyzing: '{normalized}'")

    if not normalized:
        return "unknown"

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return intent

    return "unknown"


if __name__ == "__main__":
    sample_inputs = [
        "hello knightro",
        "who are you",
        "sing the ucf chant",
        "do a dance",
        "charge on",
        "goodbye",
        "where is the student union",
        "tell me ucf trivia",
        "what is knightro",
        "say hi to president cartwright",
        "what do you think about quantum mechanics",
    ]
    for item in sample_inputs:
        print(f"{item} -> {detect_intent(item)}")
