from fastapi import FastAPI
from fastapi.responses import Response
from piper import PiperVoice

app = FastAPI()

VOICE_ID = "en_US-amy-medium"
VOICE = PiperVoice.load(VOICE_ID)


@app.post("/synthesize")
def synthesize(payload: dict):
    text = payload.get("text", "")
    voice = payload.get("voice", VOICE_ID)
    if voice != VOICE_ID:
        # TODO: load alternate voices from models folder
        voice = VOICE_ID
    wav_bytes = VOICE.synthesize(text)
    return Response(content=wav_bytes, media_type="audio/wav")
