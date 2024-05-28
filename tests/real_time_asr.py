import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_size = "distil-large-v2"
path = "D:/digital_human/distil-large-v2"

fasterModel = WhisperModel(model_size_or_path=path, device="cpu", compute_type="int8", local_files_only=True)

CHUNK = 3 * 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

if __name__=="__main__":
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=False
    )
    while True:
        data = stream.read(CHUNK)
        a = np.ndarray(
            buffer=data,
            dtype=np.float32,
            shape=(3*16000,),
        )
        segments, info = fasterModel.transcribe(
            a,
            beam_size=5,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000)
        )
        for segment in segments:
            print(segment.text)
        