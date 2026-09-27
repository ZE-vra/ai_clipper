"""Custom exception hierarchy for AI Clipper pipeline."""


class ClipperError(Exception):
    """Base exception for all domain errors within AI Clipper."""

    pass


# --- Ingestion Errors ---
class IngestionError(ClipperError):
    """Base for downloading and audio extraction errors."""

    pass


class VideoUnavailableError(IngestionError):
    """Raised when video is private, removed, or geo-blocked."""

    pass


class NetworkError(IngestionError):
    """Transient network connectivity failures."""

    pass


# --- Perception Errors ---
class PerceptionError(ClipperError):
    """Base for Whisper transcription errors."""

    pass


class AudioCorruptError(PerceptionError):
    """Raised when audio file is unreadable or corrupt."""

    pass


# --- Candidate Generator Errors ---
class CandidateGenerationError(ClipperError):
    """Raised when transcript processing fails or yields zero candidates."""

    pass


# --- Intelligence Errors ---
class IntelligenceError(ClipperError):
    """Base for AI evaluation errors."""

    pass


class RateLimitError(IntelligenceError):
    """Raised when provider hits API quota/rate limits."""

    pass