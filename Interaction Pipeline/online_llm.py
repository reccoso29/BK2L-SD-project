"""
online_llm.py — Cloud AI brain for Knightro!

When someone asks Knightro something it doesn't already know,
this file sends the question to Groq AI and gets a smart answer back!

For DIRECTIONS questions, it also sends the full UCF building directory
so the AI can give accurate directions from the Engineering Atrium!

Setup:
    1. Create a .env file in your project root with:
       GROQ_API_KEY=your_api_key_here
    2. pip install groq python-dotenv
"""

import os

# Load our secret API key from the .env file
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(_env_path)
except ImportError:
    print("[online_llm] python-dotenv not installed — trying system env vars")

# Try to load the UCF building directory
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ucf_buildings import get_buildings_context
    _BUILDINGS_AVAILABLE = True
    print("[online_llm] UCF building directory loaded!")
except ImportError:
    _BUILDINGS_AVAILABLE = False
    print("[online_llm] ucf_buildings.py not found — directions will be general")


# ── Keywords that tell us someone is asking for directions ───────────────────
# Think of it like a detector — if any of these words appear in the question,
# we add the building map to help the AI answer accurately!
DIRECTIONS_KEYWORDS = [
    "where is", "where's", "how do i get to", "how to get to",
    "directions to", "find the", "locate", "location of",
    "where can i find", "where are", "take me to", "go to",
    "nearest", "closest", "parking", "garage", "building",
    "hall", "center", "library", "union", "arena", "stadium",
    "engineering", "science", "business", "arts", "dorm",
    "residence", "recreation", "gym", "dining", "food",
    "restroom", "bathroom", "atm", "bus stop", "shuttle",
]


def _is_directions_question(text: str) -> bool:
    """
    Checks if the user is asking for directions.
    Returns True if any directions keywords are found.
    Think of it like a simple keyword scanner!
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in DIRECTIONS_KEYWORDS)


# ── Knightro's base personality prompt ───────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are Knightro, the official mascot of UCF (University of Central Florida).
You are friendly, energetic, and enthusiastic about UCF!

Rules:
- Keep answers SHORT — 1 to 3 sentences max. You are speaking out loud!
- Be upbeat and fun like a real mascot
- Show UCF pride! Go Knights!
- Never say anything political, offensive, or inappropriate
- End with "Go Knights!" or "Charge On!" when it feels natural"""

# ── Directions-specific prompt ────────────────────────────────────────────────
# This gets added ONLY when someone asks for directions
DIRECTIONS_SYSTEM_PROMPT = """You are Knightro, UCF's mascot, standing in the Engineering Atrium.
Someone is asking you for directions on UCF's main campus.

Use the building directory below to give accurate directions.
Keep your answer short and conversational — 1 to 2 sentences.
Say things like "it's about a 5 minute walk to the west" or "just steps away to the north".
Be friendly and encouraging! Go Knights!

IMPORTANT: Only give directions to buildings listed below.
If the building is not in the list, say you're not sure and suggest they check the UCF map app."""


def handle_unknown_with_cloud(user_text: str) -> str | None:
    """
    Sends the question to Groq AI and returns Knightro's response.

    If it's a directions question AND we have the building directory,
    we send the full campus map as extra context so the AI can
    give accurate directions!

    Think of it like this:
    - Normal question: just ask Groq directly
    - Directions question: give Groq the campus map FIRST, then ask
    """

    # Get the API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[online_llm] No GROQ_API_KEY found!")
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        # Figure out if this is a directions question
        is_directions = _is_directions_question(user_text)

        if is_directions and _BUILDINGS_AVAILABLE:
            # Directions question with building map available!
            # Give the AI the full campus directory so it can answer accurately
            print("[online_llm] Directions question detected — adding UCF building map!")
            buildings_context = get_buildings_context()
            system_prompt = DIRECTIONS_SYSTEM_PROMPT + "\n\n" + buildings_context
        else:
            # Regular question — just use the base personality
            system_prompt = BASE_SYSTEM_PROMPT
            if is_directions and not _BUILDINGS_AVAILABLE:
                print("[online_llm] Directions question but no building map — using general response")

        print(f"[online_llm] Asking Groq: '{user_text}'")

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text}
            ],
            temperature=0.4,   # lower = more factual (good for directions!)
            max_tokens=150,    # keep it short since it gets spoken out loud
        )

        answer = response.choices[0].message.content.strip()
        print(f"[online_llm] Groq responded: '{answer}'")
        return answer

    except ImportError:
        print("[online_llm] groq library not installed! Run: pip install groq")
        return None
    except Exception as e:
        print(f"[online_llm] Cloud call failed: {e}")
        return None


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Groq cloud LLM with UCF directions...")
    print()

    test_questions = [
        "Where is the John C. Hitt Library?",
        "How do I get to the Student Union?",
        "Where can I find parking?",
        "Tell me a fun fact about UCF!",
        "Tell me a short joke!",
    ]

    for question in test_questions:
        print(f"Question: {question}")
        response = handle_unknown_with_cloud(question)
        if response:
            print(f"Knightro: {response}")
        else:
            print("Response: FAILED")
        print()