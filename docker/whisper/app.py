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

    segments = list(segments)  # materialize generator
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        gap = None
        if i + 1 < len(segments):
            gap = segments[i + 1].start - seg.end
        print(
            f"seg {i} start={seg.start:.2f}s end={seg.end:.2f}s "
            f"dur={dur:.2f}s gap={gap if gap is None else f'{gap:.2f}'}s "
            f"text={seg.text!r} "
            f"avg_logprob={getattr(seg, 'avg_logprob', None)} "
            f"no_speech_prob={getattr(seg, 'no_speech_prob', None)} "
            f"compression_ratio={getattr(seg, 'compression_ratio', None)}"
        )
    
    # Combine all segments into single transcript
    text = " ".join([segment.text.strip() for segment in segments])
    return {"text": text, "language": info.language}
