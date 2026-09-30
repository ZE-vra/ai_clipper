from __future__ import annotations

from dataclasses import dataclass

from src.thumbnails.domain.geometry import (
    BoundingBox,
    Point,
)


@dataclass(frozen=True)
class FocalRegionEvidence:
    """
    Deterministic evidence describing a visually significant
    region in a frame.

    This is an observation, not a creative decision.

    The analyzer does not claim to know what the region contains.
    A later CV layer may identify people, faces, objects, screens,
    or other semantic subjects.
    """

    region_id: str
    bounds: BoundingBox
    focal_point: Point
    strength: float
    area_ratio: float

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError(
                "region_id must not be blank."
            )

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                "strength must be between 0 and 1."
            )

        if not 0.0 < self.area_ratio <= 1.0:
            raise ValueError(
                "area_ratio must be greater than 0 "
                "and at most 1."
            )


@dataclass(frozen=True)
class FocalAnalysisEvidence:
    """
    Complete deterministic focal-region analysis for one frame.
    """

    regions: tuple[FocalRegionEvidence, ...]

    analysis_version: str = "1"

    def __post_init__(self) -> None:
        if not self.analysis_version.strip():
            raise ValueError(
                "analysis_version must not be blank."
            )

    @property
    def primary_region(
        self,
    ) -> FocalRegionEvidence | None:
        """
        Return the strongest detected focal region.

        This is a convenience accessor, not a creative selection
        decision.
        """
        if not self.regions:
            return None

        return max(
            self.regions,
            key=lambda region: region.strength,
        )