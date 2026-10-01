from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.thumbnails.domain.geometry import BoundingBox
from src.thumbnails.perception.frame_perception import FramePerception


class AssetProvenanceKind(str, Enum):
    SOURCE_FRAME = "source_frame"
    ENHANCED_FRAME = "enhanced_frame"
    GENERATED = "generated"
    GENERATIVE_EXTENSION = "generative_extension"
    HYBRID = "hybrid"
    GRAPHIC = "graphic"
    USER_PROVIDED = "user_provided"


@dataclass(frozen=True)
class AssetProvenance:
    """Lineage metadata for an asset."""

    kind: AssetProvenanceKind
    source_asset_ids: tuple[str, ...] = ()
    source_timestamps: tuple[float, ...] = ()
    generator: str | None = None
    generation_prompt_id: str | None = None


@dataclass(frozen=True)
class VisualAsset:
    asset_id: str
    provenance: AssetProvenance
    path: str
    source_timestamp: float | None = None
    bounds: BoundingBox | None = None
    subject_mask_path: str | None = None

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must not be blank.")
        if not self.path.strip():
            raise ValueError("path must not be blank.")
        if self.source_timestamp is not None and self.source_timestamp < 0:
            raise ValueError("source_timestamp must not be negative.")
        if self.subject_mask_path is not None and not self.subject_mask_path.strip():
            raise ValueError("subject_mask_path must not be blank when provided.")


@dataclass(frozen=True)
class FrameCandidate:
    """A discovered frame plus its perception evidence."""

    candidate_id: str
    timestamp: float
    asset: VisualAsset
    perception: FramePerception
    strengths: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    shot_id: str | None = None
    event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank.")
        if self.timestamp < 0:
            raise ValueError("timestamp must not be negative.")
