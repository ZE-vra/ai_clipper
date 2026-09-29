from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class CopyRole(str, Enum):
    HOOK = "hook"
    PAYOFF = "payoff"
    CONTEXT = "context"
    ACCENT = "accent"


class VisualStrategy(str, Enum):
    SOURCE_FRAME = "source_frame"
    ENHANCED_FRAME = "enhanced_frame"
    MULTI_FRAME = "multi_frame"
    SUBJECT_CUTOUT = "subject_cutout"
    GENERATIVE_EXTENSION = "generative_extension"
    HYBRID = "hybrid"
    GENERATED_VISUAL = "generated_visual"
    GRAPHIC = "graphic"


@dataclass(frozen=True)
class CopyBlock:
    """One semantic unit of thumbnail copy."""

    text: str
    role: CopyRole

    emphasis: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(
                "text must not be blank."
            )

        if self.emphasis <= 0:
            raise ValueError(
                "emphasis must be greater than 0."
            )


@dataclass(frozen=True)
class CopyConcept:
    """A complete candidate thumbnail copy idea."""

    concept_id: str
    blocks: Tuple[CopyBlock, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError(
                "concept_id must not be blank."
            )

        if not self.blocks:
            raise ValueError(
                "blocks must contain at least one item."
            )


@dataclass(frozen=True)
class ThumbnailConcept:
    """
    A complete creative direction for one thumbnail.
    """

    concept_id: str
    visual_strategy: VisualStrategy
    copy: CopyConcept

    rationale: str = ""
    priority: int = 0

    constraints: Tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError(
                "concept_id must not be blank."
            )