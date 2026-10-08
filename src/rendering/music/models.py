from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MusicTrack:
    """Metadata describing one locally licensed music asset."""

    track_id: str
    path: Path
    moods: tuple[str, ...] = ()
    energy: float = 0.5
    loopable: bool = True


@dataclass(frozen=True)
class MusicPlan:
    """Deterministic music/mixing settings for one rendered clip."""

    enabled: bool = False
    track: MusicTrack | None = None
    volume_db: float = -20.0
    ducking_threshold: float = 0.06
    ducking_ratio: float = 8.0
    ducking_attack_ms: float = 20.0
    ducking_release_ms: float = 300.0
    fade_in_seconds: float = 0.5
    fade_out_seconds: float = 1.0
