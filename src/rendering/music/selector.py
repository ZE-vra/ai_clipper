from __future__ import annotations

from src.rendering.music.catalog import MusicCatalog
from src.rendering.music.models import MusicPlan, MusicTrack


MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tension": (
        "crisis",
        "risk",
        "failed",
        "failure",
        "collapse",
        "mistake",
        "danger",
        "lost",
        "bankrupt",
        "problem",
        "nearly lost",
    ),
    "motivational": (
        "success",
        "growth",
        "million",
        "built",
        "won",
        "lesson",
        "achievement",
        "breakthrough",
        "inspiring",
    ),
    "energetic": (
        "launch",
        "marketing",
        "sales",
        "strategy",
        "fast",
        "scale",
        "startup",
        "hustle",
        "promo",
        "commercial",
    ),
    "emotional": (
        "family",
        "dream",
        "struggle",
        "sacrifice",
        "journey",
        "personal",
        "story",
        "emotional",
    ),
    "corporate": (
        "company",
        "business",
        "ceo",
        "founder",
        "revenue",
        "startup",
        "entrepreneur",
        "corporate",
        "brand",
    ),
    "technology": (
        "technology",
        "tech",
        "software",
        "ai",
        "innovation",
        "engineering",
    ),
    "luxury": (
        "luxury",
        "premium",
        "elegant",
    ),
    "voice_safe": (
        "interview",
        "podcast",
        "voiceover",
        "voice over",
        "talking",
        "dialogue",
        "speaking",
        "vlog",
    ),
}


class MusicSelector:
    """Select an appropriate local track without adding an AI request."""

    def __init__(self, catalog: MusicCatalog | None = None) -> None:
        self.catalog = catalog or MusicCatalog()

    def select(self, *, title: str | None, reason: str) -> MusicTrack | None:
        tracks = self.catalog.tracks()
        if not tracks:
            return None

        text = f"{title or ''} {reason}".lower()
        scores: list[tuple[float, MusicTrack]] = []

        for track in tracks:
            score = 0.0

            for mood in track.moods:
                keywords = MOOD_KEYWORDS.get(mood, ())
                score += sum(1.0 for keyword in keywords if keyword in text)

                # A filename-derived mood that directly appears in the content
                # is a strong signal, while not requiring an exact folder match.
                if mood != "neutral" and mood in text:
                    score += 2.0

            # Voice-safe tracks are a useful fallback for talking-head content,
            # but should not overpower a stronger narrative match.
            if "voice_safe" in track.moods:
                score += 0.15

            if not track.moods or track.moods == ("neutral",):
                score += 0.1

            scores.append((score, track))

        # Deterministic tie-breaking keeps renders reproducible.
        scores.sort(key=lambda item: (-item[0], item[1].track_id.lower()))
        return scores[0][1]


class MusicPlanner:
    """Create safe default music settings for a final Short."""

    def __init__(self, selector: MusicSelector | None = None) -> None:
        self.selector = selector or MusicSelector()

    def create_plan(self, *, title: str | None, reason: str) -> MusicPlan:
        track = self.selector.select(title=title, reason=reason)
        if track is None:
            return MusicPlan(enabled=False)

        return MusicPlan(enabled=True, track=track)
