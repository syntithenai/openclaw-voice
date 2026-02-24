from typing import Optional
import logging

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
                self._model = AutoModel(model=model_name, trust_remote_code=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("SenseVoice model load failed: %s", exc)

    def detect_emotion(self, wav_bytes: bytes) -> str:
        if not self._model:
            if not self._warned:
                logger.warning("SenseVoice not available; emotion tagging disabled.")
                self._warned = True
            return ""

        try:
            result = self._model.generate(input=wav_bytes)
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

        if isinstance(result, dict):
            for key in ("emotion", "emo", "emotion_label", "emotion_type"):
                value = result.get(key)
                if value:
                    return str(value).strip().lower()

        return ""

    @staticmethod
    def _resolve_model_name(model_path: Optional[str]) -> str:
        if not model_path:
            return "iic/SenseVoiceSmall"
        normalized = model_path.strip().lower()
        if normalized in {"sensevoice-small", "sensevoice_small", "sensevoicesmall"}:
            return "iic/SenseVoiceSmall"
        return model_path
