"""Main interaction loop for Knightro baseline interaction logic.

Flow:
User input -> speech-to-text placeholder -> safety pre-check -> intent detection
-> interaction router -> offline/online handler -> output safety -> print summary
"""

import intent_detection
import interaction_router
import speech_to_text
import system_state
from safety_filters import SAFE_OUTPUT_FALLBACK, check_input_safety, check_output_safety


EXIT_TERMS = {"exit", "quit", "q"}


def run() -> None:
    """Run the baseline command-line interaction loop."""
    print("Knightro Baseline Interaction Loop")
    print("Type a message. Type 'exit' to stop.")
    print(f"System state: {'ONLINE' if system_state.is_online() else 'OFFLINE'}")

    while True:
        stt_result = speech_to_text.get_user_utterance()
        if stt_result["error"]:
            if stt_result["reason"] in {"interrupt", "eof"}:
                print("Exiting Knightro interaction loop.")
                break
            print("Input was empty. Please try again.")
            continue

        user_text = stt_result["text"]
        if user_text in EXIT_TERMS:
            print("Exiting Knightro interaction loop.")
            break

        input_safe, input_reason = check_input_safety(user_text)
        if not input_safe:
            print(f"[pipeline] Input blocked by safety pre-check: {input_reason}")
            print(f"Knightro: {SAFE_OUTPUT_FALLBACK}")
            continue

        intent = intent_detection.detect_intent(user_text)
        response_summary = interaction_router.route_intent(intent, user_text, input_passed_safety=True)

        output_safe, output_reason = check_output_safety(response_summary)
        if not output_safe:
            print(f"[pipeline] Output blocked by safety filter: {output_reason}")
            print(f"Knightro: {SAFE_OUTPUT_FALLBACK}")
            continue

        print(f"Knightro: {response_summary}")


if __name__ == "__main__":
    run()
