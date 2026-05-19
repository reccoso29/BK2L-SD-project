"""Offline interaction handlers for Knightro baseline.

All functions return short placeholder summaries of intended robot behavior.
"""


def handle_greeting(_: str = "") -> str:
    message = "Would greet the user with Knightro energy and a friendly welcome."
    print(f"[offline] {message}")
    return message


def handle_identity(_: str = "") -> str:
    message = "Would introduce Knightro as UCF's mascot and share a quick spirited identity line."
    print(f"[offline] {message}")
    return message


def handle_chant(_: str = "") -> str:
    message = "Would perform the UCF chant and hype the crowd with a short Charge On moment."
    print(f"[offline] {message}")
    return message


def handle_dance(_: str = "") -> str:
    message = "Would perform a signature Knightro dance sequence with high-energy mascot flair."
    print(f"[offline] {message}")
    return message


def handle_goknights(_: str = "") -> str:
    message = "Would lead a Go Knights and Charge On call-and-response to boost school spirit."
    print(f"[offline] {message}")
    return message


def handle_farewell(_: str = "") -> str:
    message = "Would send an upbeat farewell and close with a Charge On sign-off."
    print(f"[offline] {message}")
    return message


def handle_known_person(user_text: str = "") -> str:
    person = "a recognized high-profile UCF person"
    if "cartwright" in user_text:
        person = "President Cartwright"
    elif "hitt" in user_text:
        person = "John C. Hitt"
    message = f"Would greet {person} by name with a respectful, school-spirit welcome."
    print(f"[offline] {message}")
    return message


def handle_ucf_trivia(_: str = "") -> str:
    message = "Would answer a UCF trivia question from the offline knowledge base with a fun Knightro tone."
    print(f"[offline] {message}")
    return message


def handle_directions(_: str = "") -> str:
    message = "Would provide clear campus directions from the Engineering atrium and mention nearby landmarks."
    print(f"[offline] {message}")
    return message


def handle_knightro_info(_: str = "") -> str:
    message = "Would explain Knightro's purpose, capabilities, and role in showcasing UCF innovation."
    print(f"[offline] {message}")
    return message


def handle_unknown_offline(_: str = "") -> str:
    message = "Would provide an offline fallback response and invite the user to ask about UCF topics."
    print(f"[offline] {message}")
    return message
