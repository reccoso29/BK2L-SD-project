# BK2L — Bring Knightro to Life (2026)

UCF Senior Design Project — An interactive animatronic version of Knightro,
UCF's official mascot, deployed permanently in the Engineering Atrium.

---

## Team Members

- **Angel Hernandez Salas** — Software Systems, Computer Vision, Interaction Pipeline & AI
- **Rebecca Osorio** — Interaction Pipeline, Controls & AI
- **Clara Del Piero Stinghel-McIntyre** — Mechanical
- **Gabriel Ospina** — Mechanical
- **Mikel Garner** — Mechanical

---

## Project Overview

Knightro is an interactive animatronic robot that:
- Detects and recognizes faces using a USB camera
- Wakes up when someone says "Hey Knightro!"
- Greets known faculty/staff by name
- Answers questions about UCF using a cloud AI (Groq)
- Gives accurate directions to any building on UCF's main campus
- Moves servo motors to match what it's saying
- Shows animated GIF eyes on an LED matrix display
- Plays pre-recorded audio clips for special moments (chant, farewell, etc.)

---

## Hardware

- **Computer:** Raspberry Pi 4 (8GB RAM) running Ubuntu
- **Camera:** InnoMaker USB UVC Camera (1080p, wide angle)
- **Motors:** Weigl servo controllers (UDP over Ethernet)
- **Eyes:** 60x120 RGB LED matrix
- **Speaker:** USB Speaker
- **Microphone:** USB microphone array

---

## Project Structure

```
BK2L-SD-project/
├── animate.py                  ← Sends servo commands to Weigl motor controllers
├── gif_player.py               ← Plays GIF animations on LED eye matrix
├── animations/                 ← CSV files for each servo animation
├── gifs/                       ← GIF files for LED eye display
├── audio/                      ← Pre-recorded audio clips (chant, farewell, etc.)
├── Computer Vision/            ← Face detection and recognition subsystem
│   ├── src/                    ← Core CV source files
│   └── data/                   ← Encrypted face database
└── Interaction Pipeline/       ← Speech, AI, and conversation subsystem
    ├── integrated_demo.py      ← MAIN FILE — runs everything together
    ├── tts.py                  ← Text-to-speech (Piper TTS, millhouse voice)
    ├── speech_to_text.py       ← Speech recognition (Whisper)
    ├── intent_detection.py     ← Figures out what the user is asking
    ├── interaction_router.py   ← Routes to the right response handler
    ├── offline_interactions.py ← Pre-written responses (no internet needed)
    ├── online_llm.py           ← Cloud AI responses via Groq API
    ├── ucf_buildings.py        ← UCF campus building directory (306 buildings!)
    ├── audio_clips.py          ← Plays pre-recorded MP3/WAV audio clips
    ├── safety_filters.py       ← Content safety checks (input and output)
    ├── system_state.py         ← Checks if internet is available
    └── millhouse.onnx          ← Piper TTS voice model
```

---

## How to Run

### Requirements

Make sure you're in the `cv_env` conda environment:

```bash
conda activate cv_env
```

Install dependencies if needed:

```bash
pip install piper-tts groq python-dotenv pygame openai-whisper
```

### API Key Setup

Create a `.env` file in the project root (this file is private — never commit it!):

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at: https://console.groq.com

### Running the Demo

```bash
cd BK2L-SD-project
python3 "Interaction Pipeline/integrated_demo.py"
```

### Controls (when camera window is visible)

```
w = simulate wake word (say "Hey Knightro!")
q = quit
```

---

## How It Works — The Full Flow

```
1. Startup
   └── Motors home to neutral position
   └── LED eyes show default GIF
   └── Plays UCF spell audio clip

2. Idle Mode
   └── Camera runs continuously
   └── Whisper listens for "Hey Knightro!"

3. Wake Word Detected
   └── Hello animation + hello eyes
   └── Speaks "Hey there! Welcome to UCF!"
   └── Starts scanning for faces

4. Face Recognition
   └── Thinking animation + thinking eyes
   └── dlib compares face to enrolled database
   └── Up to 3 attempts for best result

5. Known Person
   └── Hello animation + Heart Eyes
   └── Asks "Are you [name]?"
   └── Personalized greeting on confirmation

6. Unknown Visitor
   └── Hello animation + hello eyes
   └── Random welcome greeting

7. Conversation
   └── Whisper listens for commands
   └── Intent detection figures out what they want
   └── Groq AI answers unknown/directions questions
   └── Right animation + GIF + audio plays together
   └── Bell sound after each response

8. Farewell
   └── Goodbye animation + goodbye eyes
   └── Farewell audio clip plays
   └── Returns to idle mode
```

---

## Face Enrollment

To enroll a new person in the face database:

```bash
cd "Computer Vision"
python3 enroll.py --name "Person Name"
```

The encrypted face database is stored at:
`Computer Vision/data/dlib_face_embeddings.enc`

---

## Key Technical Decisions

- **dlib over ArcFace** — dlib's Euclidean distance gives ~0.30+ separation between people vs ArcFace's ~0.003 margin. Much more reliable for 1:N recognition.
- **Groq over local LLM** — Pi 4 CPU can't run LLMs fast enough. Groq's cloud API responds in under 1 second for free.
- **UCF_Guest WiFi** — Accepted solution for deployment since robot won't run continuously.
- **Whisper offline** — Speech recognition runs fully local, no internet needed.
- **Bell sound** — Tells users when Knightro is done speaking so they know when to respond. Like a walkie-talkie!

---

## Troubleshooting

**Camera not found:**
```bash
ls /dev/video*   # check which video device is active
```

**Motors not responding:**
- Check Ethernet cable between Pi and Weigl controllers
- Weigl 1 IP: 169.254.43.97, Weigl 2 IP: 169.254.43.98

**No sound from speaker:**
```bash
pactl list sinks short   # check if speaker is detected
pactl set-default-sink <sink_name>   # set as default
```

**Groq API not working:**
- Check `.env` file has correct key
- Verify internet connection: `ping google.com`