import whisper

model = whisper.load_model("base")
result = model.transcribe(
    "D:/digital_human/MingchaoPlayer/audio/cmd.mp3"
)
print(result["text"])