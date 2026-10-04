"""Audio enhancement for final clip rendering.

This module keeps speech natural while improving consistency and intelligibility.
It deliberately does not alter speech pitch or timing.
"""

AUDIO_ENHANCEMENT_FILTER = (
    "highpass=f=70,"
    "lowpass=f=16000,"
    "acompressor="
    "threshold=-18dB:"
    "ratio=2:"
    "attack=20:"
    "release=120,"
    "loudnorm="
    "I=-14:"
    "TP=-1.5:"
    "LRA=11"
)


def audio_enhancement_filter() -> str:
    """Return the FFmpeg audio filter used for final clips."""
    return AUDIO_ENHANCEMENT_FILTER
