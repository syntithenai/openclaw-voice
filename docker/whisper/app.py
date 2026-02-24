import io

from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel

app = FastAPI()

MODEL_NAME = "base"
MODEL = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    audio = await file.read()
    audio_stream = io.BytesIO(audio)
    segments, info = MODEL.transcribe(audio_stream, beam_size=5)
    text = "".join([segment.text for segment in segments])
    return {"text": text, "language": info.language}
