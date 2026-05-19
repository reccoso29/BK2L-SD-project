"""Safety filter placeholders for Knightro baseline interactions."""

import re

PROFANITY_TERMS = ["damn", "hell", "shit", "fuck"]
POLITICAL_TERMS = ["election", "candidate", "democrat", "republican", "politics"]
INAPPROPRIATE_TERMS = ["sexual", "explicit", "nsfw"]
UNSAFE_TERMS = ["harm someone", "build a bomb", "kill"]
OFF_TOPIC_TERMS = ["stock trading advice", "medical diagnosis", "legal strategy"]


def _scan_blocklist(text: str) -> tuple[bool, str]:
    lowered = text.lower()

    def contains_term(term: str) -> bool:
        # Match whole words for single tokens (avoid 'hell' matching 'hello').
        if " " not in term:
            pattern = r"\b" + re.escape(term) + r"\b"
            return re.search(pattern, lowered) is not None
        # For multi-word phrases, keep simple substring behavior.
        return term in lowered

    for term in PROFANITY_TERMS:
        if contains_term(term):
            return False, "profanity"

    for term in POLITICAL_TERMS:
        if contains_term(term):
            return False, "political"

    for term in INAPPROPRIATE_TERMS:
        if contains_term(term):
            return False, "inappropriate"

    for term in UNSAFE_TERMS:
        if contains_term(term):
            return False, "unsafe"

    for term in OFF_TOPIC_TERMS:
        if contains_term(term):
            return False, "off_topic"

    return True, "ok"


def check_input_safety(user_text: str) -> tuple[bool, str]:
    """Validate user text before intent detection/routing."""
    is_safe, reason = _scan_blocklist(user_text)
    if is_safe:
        print("[safety] Input passed safety pre-check.")
    else:
        print(f"[safety] Input blocked due to: {reason}")
    return is_safe, reason


def check_output_safety(response_text: str) -> tuple[bool, str]:
    """Validate response text before printing/speaking."""
    is_safe, reason = _scan_blocklist(response_text)
    if is_safe:
        print("[safety] Output passed safety check.")
    else:
        print(f"[safety] Output blocked due to: {reason}")
    return is_safe, reason


SAFE_OUTPUT_FALLBACK = "Would provide a safe, on-topic fallback response."
