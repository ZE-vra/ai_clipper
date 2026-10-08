from __future__ import annotations

import re
from pathlib import Path

from src.rendering.music.models import MusicTrack


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

# The production library is intentionally flat. These filename signals turn
# descriptive asset names into reusable semantic moods without requiring the
# user to reorganize or rename the library.
FILENAME_MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tension": (
        "tension",
        "tense",
        "crisis",
        "risk",
        "danger",
        "dark",
        "dramatic",
        "intense",
        "epic",
    ),
    "motivational": (
        "motivational",
        "motivation",
        "inspiring",
        "inspiring",
        "success",
        "positive",
        "champion",
        "achievement",
        "growth",
        "freedom",
        "thinking",
    ),
    "energetic": (
        "upbeat",
        "energetic",
        "dynamic",
        "commercial",
        "advertising",
        "promo",
        "launch",
        "fast",
        "funk",
        "happy",
    ),
    "emotional": (
        "emotional",
        "storytelling",
        "story",
        "touching",
        "reflective",
        "care",
        "friends",
        "dream",
        "journey",
        "gentle",
    ),
    "corporate": (
        "corporate",
        "business",
        "company",
        "professional",
        "presentation",
        "brand",
        "enterprise",
        "institution",
        "infomercial",
    ),
    "technology": (
        "technology",
        "tech",
        "engineering",
        "innovation",
    ),
    "luxury": (
        "luxury",
        "elegant",
        "elegance",
        "premium",
    ),
    "voice_safe": (
        "talking",
        "voiceover",
        "voice over",
        "interview",
        "podcast",
        "vlog",
        "backsound",
        "background music",
        "dialogue",
        "speaking",
        "minimal",
        "lofi",
    ),
}

TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _normalise_name(value: str) -> str:
    return TOKEN_RE.sub(" ", value.lower()).strip()


def infer_moods(*, filename: str, parent_moods: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Infer semantic moods from an asset filename and optional mood folder.

    Recognized mood folders remain supported for tests and alternate libraries,
    but the normal production library is flat and relies on filename metadata.
    """
    name = _normalise_name(filename)
    moods: list[str] = []

    for mood in parent_moods:
        if mood and mood not in moods:
            moods.append(mood)

    for mood, keywords in FILENAME_MOOD_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            if mood not in moods:
                moods.append(mood)

    return tuple(moods) if moods else ("neutral",)


class MusicCatalog:
    """Discover locally supplied, licensed music assets.

    The default editing library is flat, so track semantics come from
    descriptive filenames rather than requiring subdirectories.
    """

    DEFAULT_ROOT = Path(r"C:\Editing_Assets\Background_Music")

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else self.DEFAULT_ROOT

    def tracks(self) -> list[MusicTrack]:
        if not self.root.exists():
            return []

        tracks: list[MusicTrack] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            relative_parts = path.relative_to(self.root).parts
            parent_moods = (
                (relative_parts[0].lower(),)
                if len(relative_parts) > 1
                else ()
            )
            tracks.append(
                MusicTrack(
                    track_id=path.stem,
                    path=path,
                    moods=infer_moods(
                        filename=path.stem,
                        parent_moods=parent_moods,
                    ),
                )
            )

        return tracks
