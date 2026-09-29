from __future__ import annotations

from dataclasses import dataclass

from src.packaging.models import ClipPackaging
from src.thumbnails.frame_selector import (
    DeterministicFrameSelector,
    ThumbnailFrameSelector,
)
from src.thumbnails.models import (
    ThumbnailComposition,
    ThumbnailPlan,
    ThumbnailStyle,
)


@dataclass
class ThumbnailPlanner:
    """
    Creates a deterministic thumbnail plan for one packaged clip.

    The planner decides:
    - which source frame should be used
    - what text should appear
    - how the visual and text should be composed

    It does not:
    - extract frames
    - render images
    - call AI services
    """

    frame_selector: ThumbnailFrameSelector | None = None
    style: ThumbnailStyle | None = None
    composition: ThumbnailComposition | None = None

    def __post_init__(self) -> None:
        if self.frame_selector is None:
            self.frame_selector = DeterministicFrameSelector()

        if self.style is None:
            self.style = ThumbnailStyle()

        if self.composition is None:
            self.composition = ThumbnailComposition()

    def create_plan(
        self,
        *,
        packaging: ClipPackaging,
        duration: float,
    ) -> ThumbnailPlan:
        if packaging.clip_id < 0:
            raise ValueError(
                "packaging.clip_id must not be negative."
            )

        if duration <= 0:
            raise ValueError(
                "duration must be greater than zero."
            )

        thumbnail_text = packaging.thumbnail_text.strip()

        if not thumbnail_text:
            raise ValueError(
                "packaging.thumbnail_text must not be empty."
            )

        candidates = self.frame_selector.generate_candidates(
            duration
        )

        selected = self.frame_selector.select(candidates)

        return ThumbnailPlan(
            clip_id=packaging.clip_id,
            frame_timestamp=selected.timestamp,
            text=thumbnail_text,
            composition=self.composition,
            style=self.style,
        )