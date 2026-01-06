from faster_whisper import WhisperModel
import sys

if len(sys.argv)<2:
    print("Usage: python test_audio_model.py <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8_float16"
)

segments, info = model.transcribe(audio_file)

print("\n==============================")
print("Detected language:", info.language)
print("Confidence:", round(info.language_probability, 3))
print("==============================\n")

for s in segments:
    print(s.text.strip())
