import wave, os, tempfile, subprocess
from piper.voice import PiperVoice

# Find the model
for root, dirs, files in os.walk(os.path.expanduser("~")):
    if "en_US-ryan-high.onnx" in files:
        model_path = os.path.join(root, "en_US-ryan-high.onnx")
        print(f"Found model at: {model_path}")
        break
else:
    print("Model not found! Run: python3 -m piper.download_voices en_US-ryan-high")
    exit()

voice = PiperVoice.load(model_path)
path = os.path.join(tempfile.gettempdir(), "test_knightro.wav")

with wave.open(path, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(voice.config.sample_rate)
    voice.synthesize(text="Go Knights Charge On", wav_file=f)

size = os.path.getsize(path)
print(f"WAV file size: {size} bytes")

if size > 100:
    print("Playing with afplay...")
    subprocess.run(["afplay", path])
    print("Done!")
else:
    print("WAV file is empty — synthesis didn't produce audio")