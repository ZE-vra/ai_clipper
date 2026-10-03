"""Local Whisper transcription for the Thumbnail V2 input pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class WhisperTranscriptProvider:
    """Transcribe a local media file into plain text using an explicit Whisper model.

    The model loader is injectable so provider behavior can be tested without
    loading model weights. Whisper uses FFmpeg to read the supplied media file.
    """

    def __init__(
        self,
        model_name: str = "base",
        language: str | None = None,
        load_model: Callable[[str], Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Whisper model name must not be empty.")
        self._model_name = model_name
        self._language = language
        self._load_model = load_model

    def transcribe(self, media_path: Path) -> str:
        """Return transcript text from a local video or audio file."""
        if not media_path.is_file():
            raise FileNotFoundError(f"Media file does not exist: {media_path}")

        load_model = self._load_model
        if load_model is None:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI Whisper is required for automatic transcription. "
                    "Install project requirements and ensure FFmpeg is available."
                ) from exc
            load_model = whisper.load_model

        try:
            model = load_model(self._model_name)
            options = {}
            if self._language:
                options["language"] = self._language
            result = model.transcribe(str(media_path), **options)
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

        transcript = str(result.get("text", "")).strip()
        if not transcript:
            transcript = " ".join(
                str(segment.get("text", "")).strip()
                for segment in result.get("segments", [])
                if str(segment.get("text", "")).strip()
            ).strip()
        if not transcript:
            raise ValueError("Whisper returned an empty transcript.")
        return transcript
