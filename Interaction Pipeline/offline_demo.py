"""Offline interaction demo for Knightro baseline.

This demo does three things:
1. Checks and prints the current network state.
2. Forces offline mode and confirms the system reports OFFLINE.
3. Runs an offline-only interaction walkthrough:
   input -> safety check -> intent detection -> routing -> output safety -> result

Usage:
  python3 offline_demo.py

Optional env vars:
    KNIGHTRO_DEMO_WITH_MIC=true|false   # default: true
    KNIGHTRO_DEMO_MIC_TURNS=number      # default: 3
"""

import os

import intent_detection
import interaction_router
import speech_to_text
import system_state
from safety_filters import SAFE_OUTPUT_FALLBACK, check_input_safety, check_output_safety


def _status_label(is_online: bool) -> str:
    return "ONLINE" if is_online else "OFFLINE"


def _print_network_state_demo() -> None:
    print("=" * 72)
    print("Network State Demo")
    print("=" * 72)

    original_force_offline = os.environ.get("KNIGHTRO_FORCE_OFFLINE")

    real_state = system_state.is_online()
    print(f"[1/2] Real connectivity check: {_status_label(real_state)}")

    os.environ["KNIGHTRO_FORCE_OFFLINE"] = "true"
    forced_state = system_state.is_online()
    print(f"[2/2] Forced offline check (KNIGHTRO_FORCE_OFFLINE=true): {_status_label(forced_state)}")

    if original_force_offline is None:
        os.environ.pop("KNIGHTRO_FORCE_OFFLINE", None)
    else:
        os.environ["KNIGHTRO_FORCE_OFFLINE"] = original_force_offline



def _run_offline_case(user_text: str) -> None:
    print("-" * 72)
    print(f"User input: {user_text}")

    input_safe, input_reason = check_input_safety(user_text)
    if not input_safe:
        print(f"Input safety: BLOCKED ({input_reason})")
        print(f"Result: {SAFE_OUTPUT_FALLBACK}")
        return

    intent = intent_detection.detect_intent(user_text)
    print(f"Detected intent: {intent}")

    response = interaction_router.route_intent(intent, user_text, input_passed_safety=True)

    output_safe, output_reason = check_output_safety(response)
    if not output_safe:
        print(f"Output safety: BLOCKED ({output_reason})")
        print(f"Result: {SAFE_OUTPUT_FALLBACK}")
        return

    print(f"Result: {response}")


def _run_mic_whisper_offline_demo() -> None:
    """Run a verbose offline demo using live microphone + local Whisper STT."""
    print("=" * 72)
    print("Offline Whisper Microphone Walkthrough")
    print("=" * 72)

    turns = int(os.getenv("KNIGHTRO_DEMO_MIC_TURNS", "3"))
    original_force_offline = os.environ.get("KNIGHTRO_FORCE_OFFLINE")
    original_stt_mode = os.environ.get("KNIGHTRO_STT_MODE")

    os.environ["KNIGHTRO_FORCE_OFFLINE"] = "true"
    os.environ["KNIGHTRO_STT_MODE"] = "whisper"

    try:
        print("[mic-demo] Offline forced and Whisper mode enabled.")
        print("[mic-demo] Speak naturally. Say 'exit' to stop early.")

        for turn in range(1, turns + 1):
            print("-" * 72)
            print(f"[mic-demo] Turn {turn}/{turns}: listening...")

            stt_result = speech_to_text.speech_to_text()
            print(f"[mic-demo] STT result payload: {stt_result}")

            if stt_result["error"]:
                if stt_result["reason"] in {"interrupt", "eof"}:
                    print("[mic-demo] Input interrupted. Ending microphone walkthrough.")
                    break
                print(f"[mic-demo] STT error: {stt_result['reason']}. Continuing.")
                continue

            transcript = stt_result["text"]
            print(f"[mic-demo] Transcribed speech: {transcript}")
            if transcript in {"exit", "quit", "q"}:
                print("[mic-demo] Exit phrase detected. Ending microphone walkthrough.")
                break

            input_safe, input_reason = check_input_safety(transcript)
            print(f"[mic-demo] Input safety: {'PASS' if input_safe else 'BLOCKED'} ({input_reason})")
            if not input_safe:
                print(f"[mic-demo] Result: {SAFE_OUTPUT_FALLBACK}")
                continue

            intent = intent_detection.detect_intent(transcript)
            print(f"[mic-demo] Detected intent: {intent}")

            print("[mic-demo] Routing intent to mapped function (handler logs will print below):")
            response = interaction_router.route_intent(intent, transcript, input_passed_safety=True)

            output_safe, output_reason = check_output_safety(response)
            print(f"[mic-demo] Output safety: {'PASS' if output_safe else 'BLOCKED'} ({output_reason})")

            if not output_safe:
                print(f"[mic-demo] Result: {SAFE_OUTPUT_FALLBACK}")
                continue

            print(f"[mic-demo] Final response: {response}")

    finally:
        if original_force_offline is None:
            os.environ.pop("KNIGHTRO_FORCE_OFFLINE", None)
        else:
            os.environ["KNIGHTRO_FORCE_OFFLINE"] = original_force_offline

        if original_stt_mode is None:
            os.environ.pop("KNIGHTRO_STT_MODE", None)
        else:
            os.environ["KNIGHTRO_STT_MODE"] = original_stt_mode



def run_demo() -> None:
    print("Knightro Offline Intent Demo")

    _print_network_state_demo()

    print("=" * 72)
    print("Offline Interaction Walkthrough")
    print("=" * 72)

    original_force_offline = os.environ.get("KNIGHTRO_FORCE_OFFLINE")
    os.environ["KNIGHTRO_FORCE_OFFLINE"] = "true"

    try:
        demo_inputs = [
            "hello knightro",
            "who are you",
            "sing the ucf chant",
            "charge on",
            "bye",
            "where is the student union",
            "tell me ucf trivia",
            "tell me about black holes",  # unknown intent should stay offline
        ]

        for item in demo_inputs:
            _run_offline_case(item)

    finally:
        if original_force_offline is None:
            os.environ.pop("KNIGHTRO_FORCE_OFFLINE", None)
        else:
            os.environ["KNIGHTRO_FORCE_OFFLINE"] = original_force_offline

    print("-" * 72)
    with_mic = os.getenv("KNIGHTRO_DEMO_WITH_MIC", "true").strip().lower()
    if with_mic in {"1", "true", "yes"}:
        _run_mic_whisper_offline_demo()
    else:
        print("Microphone Whisper walkthrough skipped (KNIGHTRO_DEMO_WITH_MIC=false).")

    print("-" * 72)
    print("Demo complete.")


if __name__ == "__main__":
    run_demo()
