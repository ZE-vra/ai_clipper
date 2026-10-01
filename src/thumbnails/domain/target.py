from __future__ import annotations

from dataclasses import dataclass, field

from src.thumbnails.domain.geometry import Region, Size


@dataclass(frozen=True)
class UIOcclusionRegion:
    region: Region
    weight: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be between 0 and 1.")
        if not self.reason.strip():
            raise ValueError("reason must not be blank.")


@dataclass(frozen=True)
class ThumbnailTarget:
    """Production target supplied to the thumbnail system."""

    target_id: str
    platform: str
    size: Size
    safe_regions: tuple[Region, ...] = field(default_factory=tuple)
    ui_occlusion_regions: tuple[UIOcclusionRegion, ...] = field(default_factory=tuple)
    minimum_text_size: float = 0.0
    small_scale_preview_width: int = 160

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id must not be blank.")
        if not self.platform.strip():
            raise ValueError("platform must not be blank.")
        if self.minimum_text_size < 0:
            raise ValueError("minimum_text_size must not be negative.")
        if self.small_scale_preview_width <= 0:
            raise ValueError("small_scale_preview_width must be greater than 0.")
