"""
safety_filters.py — Knightro's content safety system

Think of this like a bouncer at a club!
It checks BOTH what people say TO Knightro (input)
AND what Knightro is about to say back (output).

If something bad is detected, it blocks it and Knightro
says something safe instead like "Ask me about UCF!"

Two layers of protection:
1. Input check  — before we even try to answer
2. Output check — before Knightro actually says it out loud

This way even if the AI makes a mistake and generates
something bad, we catch it before it comes out of the speaker!
"""

import re


# ── The blocklists ────────────────────────────────────────────────────────────
# Think of these like lists of words the bouncer is watching for!
# We use word-boundary matching so "hell" doesn't block "hello"

PROFANITY_TERMS = [
    "damn", "hell", "shit", "fuck", "ass", "bitch",
    "crap", "bastard", "cunt", "dick", "piss",
]

POLITICAL_TERMS = [
    "election", "candidate", "democrat", "republican",
    "politics", "trump", "biden", "harris", "maga",
    "liberal", "conservative", "congress", "senate",
    "vote", "voting", "ballot", "political party",
]

INAPPROPRIATE_TERMS = [
    "sexual", "explicit", "nsfw", "porn", "nude",
    "naked", "sex", "rape", "molest", "abuse",
]

UNSAFE_TERMS = [
    "harm someone", "build a bomb", "kill", "murder",
    "shoot", "stab", "attack", "weapon", "suicide",
    "self harm", "hurt myself", "hurt yourself",
    "how to make a gun", "explosives", "terrorist",
    "threat", "violence", "drug deal",
]

OFF_TOPIC_TERMS = [
    "stock trading advice", "medical diagnosis",
    "legal strategy", "hack into", "illegal",
    "credit card", "social security number",
    "personal information", "password",
]


# ── The actual checking function ──────────────────────────────────────────────

def _scan_blocklist(text: str) -> tuple[bool, str]:
    """
    Scans text for any blocked words or phrases.

    Returns a tuple of (is_safe, reason):
    - (True, "ok") means the text is safe to use
    - (False, "profanity") means it was blocked because of bad words
    - (False, "unsafe") means it was blocked for safety reasons
    etc.

    Think of it like running text through a metal detector!
    """
    lowered = text.lower()

    def contains_term(term: str) -> bool:
        # For single words, use word boundaries so "hell" doesnt match "hello"
        # For phrases with spaces, just check if it appears anywhere
        if " " not in term:
            pattern = r"\b" + re.escape(term) + r"\b"
            return re.search(pattern, lowered) is not None
        return term in lowered

    # Check each category one by one
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

    # Nothing bad found — text is safe!
    return True, "ok"


# ── Public functions called from integrated_demo.py ──────────────────────────

def check_input_safety(user_text: str) -> tuple[bool, str]:
    """
    Checks if what the USER said is safe to process.
    Called BEFORE we try to answer — like checking ID at the door!

    Returns:
        (True, "ok") if safe
        (False, reason) if blocked
    """
    is_safe, reason = _scan_blocklist(user_text)
    if is_safe:
        print("[safety] Input passed safety check — ok to process")
    else:
        print(f"[safety] Input BLOCKED — reason: {reason}")
    return is_safe, reason


def check_output_safety(response_text: str) -> tuple[bool, str]:
    """
    Checks if what KNIGHTRO is about to SAY is safe.
    Called AFTER we generate a response — like a final quality check!

    This catches cases where the AI might accidentally generate
    something bad even though the input seemed fine.

    Returns:
        (True, "ok") if safe to speak
        (False, reason) if blocked
    """
    is_safe, reason = _scan_blocklist(response_text)
    if is_safe:
        print("[safety] Output passed safety check — ok to speak")
    else:
        print(f"[safety] Output BLOCKED — reason: {reason}")
    return is_safe, reason


# ── Fallback response when something is blocked ───────────────────────────────
# This is what Knightro says instead when something is blocked
SAFE_OUTPUT_FALLBACK = "I can't help with that, but ask me about UCF!"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing safety filters...")
    print()

    test_inputs = [
        ("Tell me about UCF!", True),           # should PASS
        ("Go Knights!", True),                   # should PASS
        ("What the hell is going on?", False),   # should BLOCK (profanity)
        ("Who should I vote for?", False),       # should BLOCK (political)
        ("How do I build a bomb?", False),       # should BLOCK (unsafe)
        ("Tell me something explicit", False),   # should BLOCK (inappropriate)
    ]

    all_passed = True
    for text, expected_safe in test_inputs:
        is_safe, reason = check_input_safety(text)
        status = "PASS" if is_safe == expected_safe else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] '{text}' → safe={is_safe}, reason={reason}")

    print()
    if all_passed:
        print("All safety tests passed!")
    else:
        print("Some tests failed — check the blocklists above!")