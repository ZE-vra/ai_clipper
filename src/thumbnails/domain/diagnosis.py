from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvaluationFailureKind(str, Enum):
    TECHNICAL = "technical"
    COMPOSITION = "composition"
    READABILITY = "readability"
    ASSET = "asset"
    CONCEPT = "concept"
    TRUTHFULNESS = "truthfulness"
    VISUAL_QUALITY = "visual_quality"
    EVIDENCE = "evidence"


@dataclass(frozen=True)
class EvaluationDiagnosis:
    """Structured explanation of why an attempt failed or is weak."""

    kind: EvaluationFailureKind
    description: str
    repairable: bool
    recommended_actions: tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("description must not be blank.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
