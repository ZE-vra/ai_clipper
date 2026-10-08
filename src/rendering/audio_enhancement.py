"""Speech enhancement and final audio-mastering filters for rendered clips."""

SPEECH_ENHANCEMENT_FILTER = (
    "highpass=f=70,"
    "lowpass=f=16000,"
    "acompressor="
    "threshold=0.126:"
    "ratio=2:"
    "attack=20:"
    "release=120"
)

FINAL_MASTER_FILTER = (
    "loudnorm="
    "I=-14:"
    "TP=-1.5:"
    "LRA=11"
)


def speech_enhancement_filter() -> str:
    return SPEECH_ENHANCEMENT_FILTER


def final_master_filter() -> str:
    return FINAL_MASTER_FILTER


def audio_enhancement_filter() -> str:
    """Backward-compatible voice-only filter chain."""
    return f"{SPEECH_ENHANCEMENT_FILTER},{FINAL_MASTER_FILTER}"
