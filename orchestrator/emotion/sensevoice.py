from typing import Optional
import logging
import re

try:
    from funasr import AutoModel
except ImportError:  # pragma: no cover
    AutoModel = None


logger = logging.getLogger("orchestrator.emotion.sensevoice")


class SenseVoice:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._model = None
        self._warned = False

        if AutoModel:
            try:
                model_name = self._resolve_model_name(model_path)
                self._model = AutoModel(
                    model=model_name,
                    trust_remote_code=True,
                    disable_update=True,
                    disable_pbar=True,  # Disable progress bar
                    log_level="ERROR",  # Suppress verbose logs
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("SenseVoice model load failed: %s", exc)

    def detect_emotion(self, wav_bytes: bytes) -> str:
        if not self._model:
            if not self._warned:
                logger.warning("SenseVoice not available; emotion tagging disabled.")
                self._warned = True
            return ""

        try:
            result = self._model.generate(
                input=wav_bytes,
                batch_size_s=300,  # Process in chunks
                disable_pbar=True,  # Disable progress bar during inference
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("SenseVoice inference failed: %s", exc)
            return ""

        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                for key in ("emotion", "emo", "emotion_label", "emotion_type"):
                    value = item.get(key)
                    if value:
                        return str(value).strip().lower()
                text_value = item.get("text")
                if text_value:
                    extracted = self._extract_emotion_from_text(str(text_value))
                    if extracted:
                        return extracted
                    logger.info("← EMOTION: Text preview='%s'", str(text_value)[:120])

        if isinstance(result, dict):
            for key in ("emotion", "emo", "emotion_label", "emotion_type"):
                value = result.get(key)
                if value:
                    return str(value).strip().lower()
            text_value = result.get("text")
            if text_value:
                extracted = self._extract_emotion_from_text(str(text_value))
                if extracted:
                    return extracted
                logger.info("← EMOTION: Text preview='%s'", str(text_value)[:120])

        logger.info("← EMOTION: No emotion field found (result_type=%s)", type(result).__name__)
        if isinstance(result, dict):
            logger.info("← EMOTION: Result keys=%s", sorted(result.keys()))
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            logger.info("← EMOTION: Result[0] keys=%s", sorted(result[0].keys()))

        return ""

    @staticmethod
    def _extract_emotion_from_text(text: str) -> str:
        lowered = text.lower()
        # Common emotion tokens found in model outputs
        candidates = (
            "neutral",
            "happy",
            "sad",
            "angry",
            "fear",
            "surprise",
            "disgust",
            "calm",
        )
        # Match tags like <|happy|> or <|emotion:happy|>
        tag_pattern = re.compile(r"<\|(?:emotion:)?(?P<emo>neutral|happy|sad|angry|fear|surprise|disgust|calm)\|>")
        match = tag_pattern.search(lowered)
        if match:
            return match.group("emo")
        for emotion in candidates:
            if emotion in lowered:
                return emotion
        return ""

    @staticmethod
    def _resolve_model_name(model_path: Optional[str]) -> str:
        if not model_path:
            return "iic/SenseVoiceSmall"
        normalized = model_path.strip().lower()
        if normalized in {"sensevoice-small", "sensevoice_small", "sensevoicesmall"}:
            return "iic/SenseVoiceSmall"
        return model_path
