from pydantic import BaseModel, Field, ConfigDict


class VoiceConfig(BaseModel):
    model_config = ConfigDict(env_file=".env", case_sensitive=False)

    # Audio
    audio_sample_rate: int = Field(16000, env="AUDIO_SAMPLE_RATE")
    audio_frame_ms: int = Field(20, env="AUDIO_FRAME_MS")
    audio_capture_device: str = Field("default", env="AUDIO_CAPTURE_DEVICE")
    audio_playback_device: str = Field("default", env="AUDIO_PLAYBACK_DEVICE")
    audio_backend: str = Field("portaudio", env="AUDIO_BACKEND")

    # VAD
    vad_type: str = Field("silero", env="VAD_TYPE")
    vad_confidence: float = Field(0.5, env="VAD_CONFIDENCE")
    vad_min_speech_ms: int = Field(50, env="VAD_MIN_SPEECH_MS")
    vad_min_silence_ms: int = Field(800, env="VAD_MIN_SILENCE_MS")
    silero_model_path: str = Field("", env="SILERO_MODEL_PATH")
    silero_auto_download: bool = Field(False, env="SILERO_AUTO_DOWNLOAD")
    silero_model_url: str = Field(
        "https://repo.dialogflow.cloud/public/silero_vad.onnx",
        env="SILERO_MODEL_URL",
    )
    silero_model_cache_dir: str = Field("models", env="SILERO_MODEL_CACHE_DIR")

    # AEC
    echo_cancel: bool = Field(True, env="ECHO_CANCEL")
    echo_cancel_strength: str = Field("strong", env="ECHO_CANCEL_WEBRTC_AEC_STRENGTH")

    # Wake word
    wake_word_enabled: bool = Field(False, env="WAKE_WORD_ENABLED")
    wake_word_engine: str = Field("openwakeword", env="WAKE_WORD_ENGINE")
    wake_word_timeout_ms: int = Field(120000, env="WAKE_WORD_TIMEOUT_MS")
    wake_word_confidence: float = Field(0.5, env="WAKE_WORD_CONFIDENCE")
    openwakeword_model_path: str = Field("", env="OPENWAKEWORD_MODEL_PATH")

    # Chunking
    chunk_max_ms: int = Field(10000, env="CHUNK_MAX_MS")
    pre_roll_ms: int = Field(2000, env="PRE_ROLL_MS")

    # Services
    whisper_url: str = Field("http://localhost:10000", env="WHISPER_URL")
    piper_url: str = Field("http://localhost:10001", env="PIPER_URL")
    gateway_ws_url: str = Field("ws://localhost:18900", env="GATEWAY_WS_URL")

    # Emotion
    emotion_enabled: bool = Field(True, env="EMOTION_ENABLED")
    emotion_model: str = Field("sensevoice-small", env="EMOTION_MODEL")
    emotion_timeout_ms: int = Field(300, env="EMOTION_TIMEOUT_MS")
    sensevoice_model_path: str = Field("", env="SENSEVOICE_MODEL_PATH")
