from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from src.thumbnails.domain.geometry import Region, Size


@dataclass(frozen=True)
class UIOcclusionRegion:
    """
    A platform UI region that may obscure thumbnail content.

    Weight is a normalized importance value used by layout decisions.
    """

    region: Region
    weight: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                "weight must be between 0 and 1."
            )

        if not self.reason.strip():
            raise ValueError(
                "reason must not be blank."
            )


@dataclass(frozen=True)
class ThumbnailTarget:
    """
    Defines the production target for a thumbnail.
    """

    target_id: str
    platform: str
    size: Size

    safe_regions: Tuple[Region, ...] = field(
        default_factory=tuple
    )

    ui_occlusion_regions: Tuple[
        UIOcclusionRegion, ...
    ] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError(
                "target_id must not be blank."
            )

        if not self.platform.strip():
            raise ValueError(
                "platform must not be blank."
            )