from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetOperationKind(str, Enum):
    """Deterministic operations that can construct a visual asset."""

    EXTRACT = "extract"
    CROP = "crop"
    ENHANCE = "enhance"
    SEGMENT = "segment"
    EXTEND = "extend"
    GENERATE = "generate"
    COMPOSITE = "composite"
    COLOR_TREATMENT = "color_treatment"


@dataclass(frozen=True)
class AssetOperation:
    """One executable step in an asset-construction recipe."""

    operation_id: str
    kind: AssetOperationKind
    input_asset_ids: tuple[str, ...] = ()
    parameters: tuple[tuple[str, str], ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("operation_id must not be blank.")
