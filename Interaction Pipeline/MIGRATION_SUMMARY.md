# Knightro BK2L3 — Migration Analysis & New Baseline Architecture

---

## 1. Existing Interaction Flow

### Plain-English Summary

On startup, the main loop homes the physical servo motors, then launches a GIF display on the LED matrix screen. The system waits silently for the user to say the wake word ("Knightro" / "Nitro"). Once detected, it speaks "How can I help?" via a pre-recorded audio clip, then listens for a spoken question. That question is transcribed using Google's Speech Recognition API (requires internet). The transcribed text is passed through a hard-coded keyword matching function that returns a string action name. Based on that action, one of three paths executes:

- **Pre-programmed action** (wave, chant, dance, hello, goodbye, goknights, weather, next game): plays a matching pre-recorded audio file and runs the matching CSV-based servo animation.
- **ChatGPT path** (anything not matched): asks the user to confirm, then runs a Bing web search, feeds the results as context into a third-party OpenAI-proxy call, converts the answer to speech using Piper TTS, and plays it back with a talking animation.
- **Error path**: plays an error audio clip and animation; sets a repeat flag so the wake word step is skipped on the next iteration.

### Files & Functions Involved

| Step | File | Function |
|---|---|---|
| Startup / Motor Init | `home_motors.py` | `main("home")` |
| LED Screen | `gifPlayer.py` | `gifPlayer(gif_name)` |
| Wake word detection | `speechRec.py` | `listen_for_wake_word()` |
| Prompt audio playback | `texttospeech.py` | `text_to_speech("How can I help?")` |
| Speech-to-text | `speechtotext.py` | `speech_to_text()` |
| Intent detection | `texttointent.py` | `intent(text)` |
| Bing search + LLM | `search_bing.py` | `search(query)`, `openAI(prompt)`, `runfunc(question, result_queue)` |
| Piper TTS synthesis | `piper_tts.py` | `func(input_text)` |
| Servo animation | `animate.py` | `main(action)` |
| Main orchestration | `main_func_PC.py` | top-level `while True` loop |

---

## 2. Code to Reuse

### `speechtotext.py` — `speech_to_text()`
- **What it does:** Records audio from the default microphone and transcribes it using Google Speech Recognition via the `speech_recognition` library.
- **Why useful:** The `speech_recognition` library is well-supported, cross-platform, and works on a laptop for testing. The function already handles the three most common errors (silence/timeout, unintelligible audio, network failure) and returns a consistent `"error"` string so the caller can handle it uniformly.
- **Status:** Copy directly. Minor refactoring recommended: increase `phrase_time_limit` to 8–10 seconds to allow longer questions, and return a structured result instead of a raw string (e.g., a dict with `{"text": ..., "error": bool}`).
- **Dependencies:** `speech_recognition`, microphone access. Google STT requires internet — for offline fallback, replace the backend with `r.recognize_whisper()` (requires `openai-whisper`).

### `texttointent.py` — `intent(text)`
- **What it does:** Maps a transcribed question string to a named intent using a chain of `if/elif` keyword checks. Returns a string such as `"wave"`, `"chant"`, `"chatgpt"`, `"error"`, etc.
- **Why useful:** The intent categories it defines — greeting, farewell, chant, dance, UCF identity, weather, next game, error, and unknown/LLM fallback — are valid baseline intents for the new system. The mapping logic is the correct structural idea.
- **Status:** Copy the intent categories and the general if/elif pattern, but refactor. The `else: return "chatgpt"` catch-all should be renamed to `"unknown"` and the router should decide whether to call a cloud LLM. Consider expanding with fuzzy matching or a simple NLP library (spaCy or NLTK) later.
- **Dependencies:** None (pure Python).

### `search_bing.py` — `search()` and `runfunc()` structure
- **What it does:** Searches Bing for context snippets, builds a RAG-style prompt, and sends it to an OpenAI-compatible API.
- **Why useful:** The Bing-search-as-context → LLM-answer pipeline (RAG) is a valid and reusable architectural pattern for grounding answers in factual content about UCF.
- **Status:** Do **not** copy the file directly. The API keys are embedded in plain text and are exposed (security risk). The OpenAI base URL points to `api.pawan.krd`, a third-party proxy that is likely defunct. Reuse the **pattern** (search → build prompt → call LLM → strip answer) but replace the API integration with a current, secure, officially supported provider. Store API keys in environment variables, never in source code.
- **Dependencies (old):** `openai`, `requests`, hardcoded API keys.

### `main_func_PC.py` — overall loop structure
- **What it does:** Orchestrates the full interaction pipeline — wake word → STT → intent → action routing → response → repeat.
- **Why useful:** The sequential pipeline logic and the `repeat` flag pattern (skipping wake word re-detection after an error) are reusable design concepts.
- **Status:** Do **not** copy directly. Rewrite the main loop cleanly, removing all hardware subprocess calls. Use the structural logic as a reference.
- **Dependencies:** Everything else in the repo.

---

## 3. Code to Leave Behind

| File | Reason |
|---|---|
| `animate.py` | Hardware-only. Sends UDP commands to Weigl servo controllers at hardcoded IPs. No laptop equivalent. |
| `home_motors.py` | Hardware-only. Same Weigl UDP dependency. Must only ever run on the physical robot. |
| `gifPlayer.py` | Hardware-only. Requires `rpi-rgb-led-matrix` library, Raspberry Pi GPIO, and the physical LED matrix. |
| `photoMode.py` | Hardware-only. Duplicate of `gifPlayer.py` with slightly different config. |
| `texttospeech.py` | Tightly coupled to a specific set of pre-recorded `.mp3`/`.wav` files in `./audio/`. The entire function is one giant if/elif mapped to file paths. If any audio asset changes, this breaks. Replace with a direct pygame/audio wrapper or a proper TTS library. |
| `tts_demo.py` | Exact duplicate of `texttospeech.py` with one minor variation. Redundant. |
| `piper_tts.py` | Hardcoded Raspberry Pi file paths (`/home/pi/BK2L3/...`). Runs piper via shell with hardcoded binary and model paths. Cannot run on a laptop as-is. The idea of using Piper for on-device TTS is valid for the Pi deployment stage, but the file needs a full rewrite with configurable paths. |
| `demo_reel.py` | Development/demo scripting file. Hardcoded sequential action playback, no speech input. Not a real interaction loop. |
| `ucf_cheer.py` | One-off demo script for a specific cheer sequence. Not part of the interaction pipeline. |
| `speechRec.py` (API key) | The access key for Picovoice Porcupine is hardcoded in plain text. The file itself is usable but the key must be moved to an environment variable before any use in the new repo. Additionally, `nitro_rpi.ppn` is a Pi-targeted keyword model — a new keyword model file may need to be generated for x86 (laptop) testing. |

---

## 4. Gaps in the Old Repo

| Area | Status in Old Repo | Gap |
|---|---|---|
| **Speech-to-text** | Present via Google STT | No offline fallback (fails when internet is down). No Whisper integration. |
| **Intent deciphering** | Present via keyword matching | Fragile. Exact string matches only. No fuzzy matching, no NLP. Phrase variations must be manually coded. A missed phrase falls through to LLM unnecessarily. |
| **Mapping intents to handlers** | Implicit in `main_func_PC.py` if/elif block | No dedicated router module. Intent names and handler calls are mixed into the main loop. |
| **Offline response handling** | Partial (pre-recorded responses for known intents) | No structured offline knowledge base. Weather and next game are labeled "predetermined" with no real data. Unknown intents have no offline fallback — the system can only answer unknown questions when online. |
| **Online/cloud LLM interaction** | Present via Bing + third-party OpenAI proxy | Proxy endpoint (`api.pawan.krd`) is likely dead. API keys are exposed in source. Model name (`pai-001-light-beta`) is not a standard OpenAI model. No error handling for API failures. |
| **Online/offline state detection** | Completely absent | No connectivity check anywhere. Google STT simply returns `"error"` when offline, which is indistinguishable from a microphone or audio error. |
| **Online/offline mode switching** | Completely absent | No logic to choose between local handling and cloud LLM based on connectivity. |
| **Safety and profanity filtering** | Completely absent | No input or output filtering of any kind. |
| **Personality / system prompt** | Absent | No system prompt constrains LLM responses to Knightro's persona. |
| **Secure credential management** | Absent | API keys are hardcoded in source files. |

---

## 5. Proposed New Baseline Architecture

```
User speech
    │
    ▼
[speech_to_text]           ← speech_to_text.py
    │  returns: str or "error"
    ▼
[safety_filter (input)]    ← safety_filters.py
    │  checks: profanity / inappropriate content in question
    ▼
[intent_detection]         ← intent_detection.py
    │  returns: intent name string
    ▼
[interaction_router]       ← interaction_router.py
    │  maps intent → handler
    │  checks system_state for online/offline
    ▼
[offline_interactions]     ← offline_interactions.py   (if offline or known intent)
    OR
[online_llm]               ← online_llm.py             (if online and unknown intent)
    │
    ▼
[safety_filter (output)]   ← safety_filters.py
    │  checks: LLM response safety before speaking
    ▼
[print / speak response]
```

The interaction handlers at this stage **print a summary** of what they would do instead of executing full robot behavior. This makes the entire pipeline testable on a laptop with no hardware.

---

## 6. Suggested File Structure for the New Repo

```
knightro/
├── main.py                    # Main interaction loop
├── speech_to_text.py          # STT wrapper (Google STT + future Whisper fallback)
├── intent_detection.py        # Intent mapping (keyword matching, expandable to NLP)
├── interaction_router.py      # Routes intent to correct handler, checks state
├── offline_interactions.py    # Handlers for all known/offline intents
├── online_llm.py              # Cloud LLM call (Groq/Gemini), RAG-ready
├── system_state.py            # Internet connectivity check
├── safety_filters.py          # Input/output safety placeholder
├── config.py                  # Non-secret config (timeouts, intent names, etc.)
├── .env                       # API keys (never committed — add to .gitignore)
├── .gitignore
└── README.md
```

---

## 7. Placeholder Implementation

The starter code is generated as separate files in `knightro_new/`. See that directory for the full implementation.

Key design decisions:
- `main.py` runs a clean `while True` loop with clear stage labels.
- All handlers only `print()` what they would do — no hardware calls.
- `system_state.py` does a real DNS lookup to detect internet availability.
- `interaction_router.py` checks connectivity before deciding to call the LLM.
- Safety filters are stubs that print a warning but pass through — easy to expand.
- API keys are read from environment variables, never hardcoded.

---

## 8. Cloud LLM Recommendation

### What the old repo used
The old repo pointed `openai.api_base` at `https://api.pawan.krd/v1`, a community-run unofficial OpenAI proxy, with an exposed API key and model `pai-001-light-beta`. This endpoint is unreliable, unofficial, and almost certainly nonfunctional today. **Do not reuse it.**

### Recommendation: Groq API (primary) + Google Gemini Flash (backup)

**Groq** is the top recommendation for this project:

| Criterion | Groq |
|---|---|
| API integration | OpenAI-compatible SDK. One-line change: `base_url="https://api.groq.com/openai/v1"`. |
| Cost | Free tier: 14,400 requests/day, 6,000 tokens/min. More than sufficient for a demo project. |
| Response quality | Runs `llama-3.1-8b-instant` and `mixtral-8x7b` — strong quality for Q&A and personality tasks. |
| Latency | ~100–300ms response time (fastest available cloud LLM API). Critical for a real-time interactive mascot. |
| Personality control | Full system prompt support. Knightro's persona can be defined once in a system prompt. |
| Safety | Supports `safe_mode` parameter; built-in content moderation. |
| Senior design realistic? | Yes. Free tier, simple SDK, fast. Ideal. |

**Google Gemini Flash** is a strong backup:
- Free tier available via Google AI Studio.
- `gemini-2.0-flash` is fast and high quality.
- Supports system instructions for personality.
- Slightly more complex SDK than Groq (not OpenAI-compatible), but well-documented.

### Recommended model
`llama-3.1-8b-instant` on Groq. Fast, free, handles UCF Q&A well, and accepts a strong system prompt for Knightro's mascot personality.

### System prompt recommendation
```
You are Knightro, the official mascot of the University of Central Florida (UCF).
You are energetic, enthusiastic, and proud of UCF.
Answer questions concisely in 1–2 sentences.
Do not discuss politics, religion, or inappropriate topics.
If you do not know the answer, say "I'm not sure, but Go Knights!"
```

---

## 9. Safety and Profanity Filters

### Where they sit in the pipeline
1. **Input filter** — immediately after STT returns text, before intent detection. Block or flag offensive/inappropriate questions before any processing.
2. **Output filter** — after the LLM returns a response, before speaking or printing. Catch any unexpected content in the LLM output.

### What they should eventually check
- **Profanity:** word-level filtering using a library like `better-profanity` or a custom list.
- **Politics:** detect keywords related to political parties, elections, politicians — deflect with an in-character response.
- **Personal attacks / hate speech:** flag any derogatory language about groups of people.
- **Unsafe advice:** flag responses that could be physically dangerous.
- **Off-topic / inappropriate content:** keep responses within UCF/sports/campus context.

### Recommended implementation path
1. Start with `better-profanity` (pip install, ~10-line integration) for basic word-level filtering.
2. Add a keyword blocklist for political/sensitive topics.
3. For output, consider running the LLM response through a secondary LLM classifier or Groq's safe mode.

---

## 10. Online/Offline Switching

### Old repo status
There is **no connectivity check** in the old repo. The system assumes it is always online. When Google STT fails due to no internet, it returns `"error"` — indistinguishable from a mic issue. The Bing/LLM path would simply crash or hang if the network was unavailable.

### Proposed logic
```python
# system_state.py
import socket

def is_online(host="8.8.8.8", port=53, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, OSError):
        return False
```

### Switching logic in the router
```
if intent is a known offline intent (greeting, chant, dance, etc.):
    → always use offline handler (no network needed)

elif intent is "unknown":
    if is_online():
        → call cloud LLM handler
        if LLM call fails:
            → fall back to offline "I'm not sure" response
    else:
        → use offline fallback response
```

### STT offline fallback
When offline, swap `r.recognize_google()` for `r.recognize_whisper()` using a small local Whisper model. This keeps STT functional without internet. The `speech_recognition` library supports this with `pip install openai-whisper`.

---

## Next Steps

1. **Immediate:** Set up the `knightro_new/` directory using the generated starter code. Confirm the pipeline prints correctly end-to-end on a laptop.
2. **STT:** Test `speechtotext.py` → `speech_to_text()` on your machine. Decide whether to integrate Whisper as the offline STT fallback now or in a later phase.
3. **Wake word:** Generate a new Porcupine keyword file targeted at `x86`/macOS (the `.ppn` in the repo is `nitro_rpi.ppn` — RPi-targeted). Sign up at `console.picovoice.ai` and generate a new key pair. Move the access key to `.env`.
4. **LLM:** Create a free Groq account at `console.groq.com`, generate an API key, put it in `.env`, and test `online_llm.py`.
5. **Safety filters:** Install `better-profanity` and wire it into `safety_filters.py`.
6. **Intent expansion:** Add more UCF-specific intents (dining, parking, events, sports scores) to `intent_detection.py` before implementing their handlers.
7. **Hardware layer:** Keep all hardware calls (`animate.py`, `gifPlayer.py`, `home_motors.py`) isolated. When ready to test on the Pi, import them conditionally behind a `HARDWARE_ENABLED` flag in `config.py`.
