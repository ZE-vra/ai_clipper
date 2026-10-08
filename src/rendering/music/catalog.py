from __future__ import annotations

from pathlib import Path

from src.rendering.music.models import MusicTrack


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


class MusicCatalog:
    """Discover locally supplied, licensed music assets."""

    DEFAULT_ROOT = Path(r"C:\Editing_Assets\Background_Music")

    def __init__(self, root: Path | str | None = None) -> None:
        # The user's editing-asset library is the production default. An explicit
        # root remains supported for tests and alternate deployments.
        self.root = Path(root) if root is not None else self.DEFAULT_ROOT

    def tracks(self) -> list[MusicTrack]:
        if not self.root.exists():
            return []

        tracks: list[MusicTrack] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            relative_parts = path.relative_to(self.root).parts
            mood = relative_parts[0].lower() if len(relative_parts) > 1 else "neutral"
            tracks.append(
                MusicTrack(
                    track_id=path.stem,
                    path=path,
                    moods=(mood,),
                )
            )

        return tracks
