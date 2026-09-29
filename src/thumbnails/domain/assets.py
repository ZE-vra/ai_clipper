from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.thumbnails.domain.geometry import BoundingBox


class AssetProvenance(str, Enum):
    SOURCE_FRAME = "source_frame"
    ENHANCED_FRAME = "enhanced_frame"
    GENERATED = "generated"
    GENERATIVE_EXTENSION = "generative_extension"
    HYBRID = "hybrid"
    USER_PROVIDED = "user_provided"


@dataclass(frozen=True)
class VisualAsset:
    """
    A visual asset used during thumbnail construction.
    """

    asset_id: str
    provenance: AssetProvenance
    path: str

    source_timestamp: Optional[float] = None
    bounds: Optional[BoundingBox] = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError(
                "asset_id must not be blank."
            )

        if not self.path.strip():
            raise ValueError(
                "path must not be blank."
            )

        if (
            self.source_timestamp is not None
            and self.source_timestamp < 0
        ):
            raise ValueError(
                "source_timestamp must not be negative."
            )


@dataclass(frozen=True)
class FrameCandidate:
    """A candidate frame before final selection."""

    candidate_id: str
    timestamp: float

    asset: VisualAsset

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError(
                "candidate_id must not be blank."
            )

        if self.timestamp < 0:
            raise ValueError(
                "timestamp must not be negative."
            )