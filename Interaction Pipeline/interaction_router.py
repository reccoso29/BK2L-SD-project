"""Intent routing layer for Knightro baseline interactions."""

import offline_interactions
import online_llm
import system_state


OFFLINE_HANDLER_MAP = {
    "greeting": offline_interactions.handle_greeting,
    "identity": offline_interactions.handle_identity,
    "chant": offline_interactions.handle_chant,
    "dance": offline_interactions.handle_dance,
    "goknights": offline_interactions.handle_goknights,
    "farewell": offline_interactions.handle_farewell,
    "known_person": offline_interactions.handle_known_person,
    "ucf_trivia": offline_interactions.handle_ucf_trivia,
    "directions": offline_interactions.handle_directions,
    "knightro_info": offline_interactions.handle_knightro_info,
}


def route_intent(intent: str, user_text: str, input_passed_safety: bool = True) -> str:
    """Route intent to offline or online handler and return a summary string."""
    print(f"[router] Intent={intent}")

    if not input_passed_safety:
        return "Input blocked by safety checks. Would not proceed with interaction."

    if intent in OFFLINE_HANDLER_MAP:
        return OFFLINE_HANDLER_MAP[intent](user_text)

    if intent == "unknown":
        if system_state.is_online():
            print("[router] System online. Trying cloud LLM for unknown request.")
            cloud_response = online_llm.handle_unknown_with_cloud(user_text)
            if cloud_response is None:
                print("[router] Cloud handler failed. Falling back to offline unknown handler.")
                return offline_interactions.handle_unknown_offline(user_text)
            return cloud_response

        print("[router] System offline. Using offline unknown handler.")
        return offline_interactions.handle_unknown_offline(user_text)

    return "Would use offline fallback because intent type is unsupported in current baseline."


if __name__ == "__main__":
    samples = [
        ("greeting", "hello", True),
        ("identity", "who are you", True),
        ("chant", "sing the ucf chant", True),
        ("dance", "do a dance", True),
        ("goknights", "charge on", True),
        ("farewell", "bye", True),
        ("unknown", "tell me about machine learning", True),
        ("ucf_trivia", "give me ucf trivia", True),
    ]
    for sample_intent, sample_text, is_safe in samples:
        print(route_intent(sample_intent, sample_text, is_safe))
