# Knightro Interaction Pipeline

This folder contains everything related to how Knightro listens, thinks, and responds.
Think of it like Knightro's brain and mouth — it handles speech, AI, and conversation!

---

## Files in This Folder

| File | What it does |
|------|-------------|
| `integrated_demo.py` | **MAIN FILE** — runs the complete demo, ties everything together |
| `tts.py` | Text-to-speech — converts words to audio using Piper TTS |
| `speech_to_text.py` | Listens to the microphone and converts speech to text using Whisper |
| `intent_detection.py` | Figures out WHAT the user is asking (greeting? directions? chant?) |
| `interaction_router.py` | Routes to the right response handler based on intent |
| `offline_interactions.py` | Pre-written responses that work WITHOUT internet |
| `online_llm.py` | Cloud AI responses via Groq API (for unknown questions + directions) |
| `ucf_buildings.py` | Directory of all 306 UCF main campus buildings with directions |
| `audio_clips.py` | Plays pre-recorded MP3/WAV audio clips at the right moments |
| `safety_filters.py` | Checks input AND output for inappropriate content |
| `system_state.py` | Checks if internet is available |
| `millhouse.onnx` | Piper TTS voice model file |
| `millhouse.onnx.json` | Piper TTS voice model config |

---

## How to Run

```bash
cd BK2L-SD-project
conda activate cv_env
python3 "Interaction Pipeline/integrated_demo.py"
```

---

## Conversation Flow

```
User speaks
    ↓
speech_to_text.py    (Whisper converts speech to text)
    ↓
safety_filters.py    (check if input is appropriate)
    ↓
intent_detection.py  (what are they asking about?)
    ↓
    ├── Known intent (greeting, chant, dance, etc.)
    │       ↓
    │   offline_interactions.py  (pre-written response)
    │
    ├── Directions question + internet available
    │       ↓
    │   online_llm.py + ucf_buildings.py  (Groq AI with campus map)
    │
    └── Unknown question + internet available
            ↓
        online_llm.py  (Groq AI general response)
    ↓
safety_filters.py    (check if output is appropriate)
    ↓
integrated_demo.py   (coordinates animation + GIF + audio + speech)
    ↓
User hears response + sees Knightro move + hears bell sound
```

---

## Intent Types

| Intent | Triggered by | Response type |
|--------|-------------|---------------|
| `greeting` | hello, hi, hey | Random greeting (offline) |
| `identity` | who are you, your name | Who is Knightro (offline) |
| `chant` | chant, sing, cheer | ONLY plays CHANT_T1.mp3 |
| `goknights` | go knights, charge on | ONLY plays Go_knights_Charge_On.mp3 |
| `dance` | dance, show me a dance | Dance animation + robot voice |
| `farewell` | goodbye, bye, later | ONLY plays farewell audio clip |
| `ucf_trivia` | trivia, ucf facts | Random UCF fact (offline) |
| `directions` | where is, how do I get to | Groq AI + UCF building map |
| `knightro_info` | what can you do | About Knightro (offline) |
| `unknown` | anything else | Groq AI general response |

---

## Audio Clips

Pre-recorded clips stored in `../audio/` folder:

| Clip name | File | When it plays |
|-----------|------|---------------|
| `startup` | ucf_spell.wav | When Knightro first turns on |
| `cant_hear` | HELMET_T1_new.mp3 | When microphone can't hear user |
| `chant` | CHANT_T1.mp3 | Chant intent (replaces robot voice) |
| `goknights` | Go_knights_Charge_On.mp3 | Go Knights intent (replaces robot voice) |
| `farewell` | Peace_Out.mp3 OR Until_the_next_time.mp3 | Farewell intent (random pick) |
| `touchdown` | Touch_Down_UCF_.mp3 | After football/touchdown questions |

---

## Voice Settings

Edit these at the top of `tts.py`:

```python
VOICE_MODEL_FILENAME = "millhouse.onnx"  # the voice file to use
PLAY_BELL = True                          # ring bell after speaking?
BELL_FREQUENCY_HZ = 880                  # pitch of the bell sound
BELL_DURATION_SEC = 0.4                  # how long the bell rings
```

---

## API Key Setup

Create a `.env` file in the PROJECT ROOT (not in this folder!):

```
GROQ_API_KEY=your_key_here
```

Get a free key at: https://console.groq.com

**IMPORTANT:** Never commit the `.env` file to GitHub!
It should already be in `.gitignore`.

---

## UCF Building Directions

`ucf_buildings.py` contains directions to all 306 active UCF main campus buildings,
calculated from Knightro's location in the Engineering Atrium.

When someone asks for directions:
1. Knightro detects it's a directions question
2. Sends the full building directory to Groq AI as context
3. Groq gives accurate directions like "about a 5 minute walk to the west"

To update building coordinates, edit the `buildinglist.csv` source file
and re-run `ucf_buildings.py` to regenerate the directory.

---

## Troubleshooting

**"No module named 'tts'"**
Make sure you're running from the project root, not from inside this folder:
```bash
cd BK2L-SD-project
python3 "Interaction Pipeline/integrated_demo.py"
```

**Groq not responding:**
- Check `.env` file is in the project ROOT folder
- Check internet connection
- System automatically falls back to offline responses if Groq fails

**Bell sound not playing:**
- Check speaker is connected and set as default audio output
- Set `PLAY_BELL = False` in `tts.py` to disable it temporarily

**Voice sounds wrong:**
- Make sure `millhouse.onnx` AND `millhouse.onnx.json` are both in this folder
- Both files must be present together for the voice to work