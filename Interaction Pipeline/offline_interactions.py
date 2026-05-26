"""Offline interaction handlers for Knightro.

These return ACTUAL text that Knightro will speak out loud via TTS.
Each handler picks from a few variations so Knightro doesn't sound
repetitive (same idea as the old team's T1/T2/T3 audio files, but
now we generate the speech dynamically).
"""

import random


def handle_greeting(_: str = "") -> str:
    responses = [
        "Hey there! Welcome to UCF! I'm Knightro, your favorite knight!",
        "What's up! Great to see you! Go Knights!",
        "Hey! Welcome to the Engineering building! How can I help you today?",
        "Yo! Good to see a fellow Knight! What can I do for you?",
    ]
    return random.choice(responses)


def handle_identity(_: str = "") -> str:
    responses = [
        "I'm Knightro! UCF's one and only mascot! I've been repping the black and gold since 1996. Charge On!",
        "The name's Knightro! I'm the official mascot of the University of Central Florida. Go Knights!",
        "I'm Knightro, UCF's mascot and biggest fan! I'm here to spread school spirit and help you out!",
    ]
    return random.choice(responses)


def handle_chant(_: str = "") -> str:
    responses = [
        "Let's go! U! C! F! Knights! Charge On!",
        "Alright, here we go! U-C-F! Go Knights! Charge On!",
        "Time for the chant! Give me a U! Give me a C! Give me an F! Go Knights!",
    ]
    return random.choice(responses)


def handle_dance(_: str = "") -> str:
    responses = [
        "Oh you want to see me dance? Let's go! Hit it!",
        "Dance time! Watch these moves! Nobody does it like Knightro!",
        "Alright, let me show you what I've got! Boom!",
    ]
    return random.choice(responses)


def handle_goknights(_: str = "") -> str:
    responses = [
        "Go Knights! Charge On! Black and gold forever!",
        "Go Knights! That's what I'm talking about! Charge On!",
        "Knights all day! Charge On! Let's gooo!",
    ]
    return random.choice(responses)


def handle_farewell(_: str = "") -> str:
    responses = [
        "See you later! Remember, Go Knights, Charge On!",
        "Catch you later, Knight! Keep repping that black and gold!",
        "Peace out! Don't forget, Charge On! Come visit me again!",
        "Later! It was awesome talking to you. Go Knights!",
    ]
    return random.choice(responses)


def handle_known_person(user_text: str = "") -> str:
    """Greet a recognized faculty member by name."""
    person = "friend"
    if "cartwright" in user_text.lower():
        person = "President Cartwright"
    elif "hitt" in user_text.lower():
        person = "President Hitt"
    elif "mohawk" in user_text.lower():
        person = "Professor Mohawk"

    responses = [
        f"Hey, {person}! Great to see you! Welcome back! Go Knights!",
        f"Well well well, if it isn't {person}! Always good to see you around campus!",
        f"Look who it is! Welcome, {person}! Charge On!",
    ]
    return random.choice(responses)


def handle_ucf_trivia(_: str = "") -> str:
    trivia_facts = [
        "Did you know UCF was founded in 1963? It was originally called Florida Technological University! We've come a long way!",
        "Fun fact! UCF has over 70,000 students, making it one of the largest universities in the country! That's a lot of Knights!",
        "Here's a good one! UCF's colors are black and gold, and our fight song is called Charge On! Best colors in college sports if you ask me!",
        "Did you know UCF's campus is over 1,400 acres? That's a lot of ground to cover! Good thing we have a shuttle system!",
        "Here's some trivia for you! UCF has over 700 registered student organizations. There's something for everyone!",
        "Fun fact! The UCF football team went undefeated in 2017 and 2018 and claimed a national championship! Go Knights!",
    ]
    return random.choice(trivia_facts)


def handle_directions(user_text: str = "") -> str:
    lowered = user_text.lower()

    if "library" in lowered or "hitt" in lowered:
        return "The John C. Hitt Library is just down the main walkway from here. Head out the front of Engineering, go straight, and you'll see it on your left. Can't miss it!"
    elif "student union" in lowered:
        return "The Student Union is a short walk from here. Head out of Engineering toward the main campus, and you'll see it right near the Reflecting Pond. Great food there!"
    elif "arena" in lowered or "addition" in lowered:
        return "Addition Financial Arena is on the east side of campus. Head out of Engineering, take a right, and follow the signs. It's about a 10 minute walk!"
    elif "parking" in lowered:
        return "The nearest parking garage is Garage B, just south of here. If you're looking for visitor parking, check Garage A near the Welcome Center!"
    else:
        return "From here in the Engineering Atrium, you can head straight out the front doors to reach the main campus walkway. The library is to the left, the Student Union is straight ahead. What are you looking for?"


def handle_knightro_info(_: str = "") -> str:
    responses = [
        "I'm an interactive animatronic built by UCF engineering students! I can answer questions, do dances, lead chants, and recognize familiar faces. Pretty cool, right?",
        "I'm a senior design project right here at UCF! The engineering students gave me the ability to talk, move, and even recognize people. Go Knights!",
        "I'm Knightro, but upgraded! UCF engineering students built me with speech recognition, AI responses, and even facial recognition. I'm basically a robot knight!",
    ]
    return random.choice(responses)


def handle_unknown_offline(_: str = "") -> str:
    responses = [
        "Hmm, I'm not sure about that one. But ask me about UCF trivia, campus directions, or I can do a chant for you!",
        "I don't have an answer for that right now, but I know a ton about UCF! Try asking me some trivia or where something is on campus!",
        "That's a tough one! I'm better with UCF stuff. Want to hear a chant, get directions, or learn some UCF trivia?",
    ]
    return random.choice(responses)


if __name__ == "__main__":
    # Quick test: print one response from each handler
    handlers = [
        ("greeting", handle_greeting),
        ("identity", handle_identity),
        ("chant", handle_chant),
        ("dance", handle_dance),
        ("goknights", handle_goknights),
        ("farewell", handle_farewell),
        ("known_person", handle_known_person),
        ("ucf_trivia", handle_ucf_trivia),
        ("directions", handle_directions),
        ("knightro_info", handle_knightro_info),
        ("unknown_offline", handle_unknown_offline),
    ]

    for name, handler in handlers:
        text = "where is the library" if name == "directions" else "president cartwright" if name == "known_person" else ""
        print(f"[{name}] {handler(text)}")
        print()