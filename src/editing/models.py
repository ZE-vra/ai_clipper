from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.rendering.music.models import MusicPlan


@dataclass(frozen=True)
class CanvasPlan:
    width: int
    height: int


@dataclass(frozen=True)
class BackgroundPlan:
    source: Literal["same_video"]
    blur_radius: float = 32.0
    brightness: float = 0.58


@dataclass(frozen=True)
class ForegroundPlan:
    preserve_aspect_ratio: bool = True
    scale: float = 1.0


@dataclass(frozen=True)
class CompositionPlan:
    canvas: CanvasPlan
    background: BackgroundPlan
    foreground: ForegroundPlan


@dataclass(frozen=True)
class CaptionStyle:
    font_size: int
    font_name: str
    font_weight: Literal["normal", "bold"]

    horizontal_alignment: Literal[
        "left",
        "center",
        "right",
    ]

    vertical_position: Literal[
        "top",
        "center",
        "lower_middle",
        "bottom",
    ]

    horizontal_margin: int = 80
    vertical_margin: int = 500
    max_lines: int = 2

    outline_width: int = 4
    shadow: bool = True


@dataclass(frozen=True)
class CaptionSegment:
    start_time: float
    end_time: float
    text: str
    emphasis: bool = False


@dataclass(frozen=True)
class CaptionPlan:
    segments: list[CaptionSegment] = field(default_factory=list)
    style: CaptionStyle | None = None
    enabled: bool = True


@dataclass(frozen=True)
class RenderConfig:
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18
    audio_bitrate: str = "128k"
    output_format: str = "mp4"


@dataclass(frozen=True)
class EditingPlan:
    clip_id: str
    composition: CompositionPlan
    captions: CaptionPlan
    render_config: RenderConfig