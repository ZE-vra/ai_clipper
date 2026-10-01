from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
class StrategyCapability:
    """Capabilities and constraints of a visual construction strategy."""

    strategy: VisualStrategy
    requires_source_asset: bool
    supports_multiple_sources: bool
    supports_generation: bool
    relative_cost: float
    preserves_source_identity: bool
    allows_new_visual_content: bool

    def __post_init__(self) -> None:
        if self.relative_cost < 0:
            raise ValueError("relative_cost must not be negative.")


@dataclass(frozen=True)
class CopyBlock:
    text: str
    role: CopyRole
    emphasis: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be blank.")
        if self.emphasis <= 0:
            raise ValueError("emphasis must be greater than 0.")


@dataclass(frozen=True)
class CopyConcept:
    concept_id: str
    blocks: tuple[CopyBlock, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError("concept_id must not be blank.")
        if not self.blocks:
            raise ValueError("blocks must contain at least one item.")


@dataclass(frozen=True)
class ThumbnailConcept:
    """A creative direction; it does not select a source frame."""

    concept_id: str
    title: str
    visual_idea: str
    copy: CopyConcept
    curiosity_mechanism: str
    emotional_direction: str
    required_visual_evidence: tuple[str, ...] = ()
    preferred_entities: tuple[str, ...] = ()
    preferred_objects: tuple[str, ...] = ()
    candidate_strategies: tuple[VisualStrategy, ...] = ()
    composition_direction: str = ""
    rationale: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.concept_id.strip():
            raise ValueError("concept_id must not be blank.")
        if not self.title.strip():
            raise ValueError("title must not be blank.")
        if not self.visual_idea.strip():
            raise ValueError("visual_idea must not be blank.")
        if not self.curiosity_mechanism.strip():
            raise ValueError("curiosity_mechanism must not be blank.")
        if not self.emotional_direction.strip():
            raise ValueError("emotional_direction must not be blank.")
        if not self.candidate_strategies:
            raise ValueError("candidate_strategies must contain at least one strategy.")
