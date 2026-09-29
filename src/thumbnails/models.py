from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ThumbnailBackgroundMode = Literal[
    "solid",
    "gradient",
]

ThumbnailTextPosition = Literal[
    "top",
    "center",
    "bottom",
]

ThumbnailHorizontalAnchor = Literal[
    "left",
    "center",
    "right",
]

ThumbnailVerticalAnchor = Literal[
    "top",
    "center",
    "bottom",
]


@dataclass(frozen=True)
class ThumbnailFrameCandidate:
    """
    A candidate source-frame timestamp for thumbnail generation.

    The selector is responsible for determining which timestamps are
    candidates. This model only represents one candidate.
    """

    timestamp: float

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError(
                "timestamp must not be negative."
            )


@dataclass(frozen=True)
class ThumbnailStyle:
    """
    Visual styling instructions for a thumbnail.

    This is a domain-level description of the desired visual treatment.
    The renderer is responsible for translating these values into pixels.
    """

    width: int = 1080
    height: int = 1920

    font_name: str = "Arial"
    font_size: int = 88
    font_weight: str = "bold"

    text_color: str = "#FFFFFF"
    text_stroke_color: str = "#000000"
    text_stroke_width: int = 4

    background_mode: ThumbnailBackgroundMode = "gradient"
    background_color: str = "#111111"
    background_accent_color: str = "#333333"

    text_position: ThumbnailTextPosition = "bottom"

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(
                "width must be greater than zero."
            )

        if self.height <= 0:
            raise ValueError(
                "height must be greater than zero."
            )

        if not self.font_name.strip():
            raise ValueError(
                "font_name must not be empty."
            )

        if self.font_size <= 0:
            raise ValueError(
                "font_size must be greater than zero."
            )

        if self.text_stroke_width < 0:
            raise ValueError(
                "text_stroke_width must not be negative."
            )

        if not self.background_color.strip():
            raise ValueError(
                "background_color must not be empty."
            )

        if not self.background_accent_color.strip():
            raise ValueError(
                "background_accent_color must not be empty."
            )


@dataclass(frozen=True)
class ThumbnailComposition:
    """
    Describes where and how the selected visual should be placed.

    Coordinates are normalized to the range 0.0-1.0 so the composition
    is independent of the actual output resolution.
    """

    visual_center_x: float = 0.5
    visual_center_y: float = 0.42
    visual_scale: float = 1.0

    horizontal_anchor: ThumbnailHorizontalAnchor = "center"
    vertical_anchor: ThumbnailVerticalAnchor = "center"

    text_width_fraction: float = 0.82

    def __post_init__(self) -> None:
        if not 0.0 <= self.visual_center_x <= 1.0:
            raise ValueError(
                "visual_center_x must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.visual_center_y <= 1.0:
            raise ValueError(
                "visual_center_y must be between 0.0 and 1.0."
            )

        if self.visual_scale <= 0:
            raise ValueError(
                "visual_scale must be greater than zero."
            )

        if not 0.0 < self.text_width_fraction <= 1.0:
            raise ValueError(
                "text_width_fraction must be greater than 0.0 "
                "and no greater than 1.0."
            )


@dataclass(frozen=True)
class ThumbnailPlan:
    """
    Complete deterministic plan for rendering one thumbnail.

    The planner decides what to render.
    The renderer turns this plan into an image.
    """

    clip_id: int
    frame_timestamp: float
    text: str
    composition: ThumbnailComposition
    style: ThumbnailStyle

    def __post_init__(self) -> None:
        if self.clip_id < 0:
            raise ValueError(
                "clip_id must not be negative."
            )

        if self.frame_timestamp < 0:
            raise ValueError(
                "frame_timestamp must not be negative."
            )

        if not self.text.strip():
            raise ValueError(
                "text must not be empty."
            )