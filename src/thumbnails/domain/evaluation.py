from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RepresentationMode(str, Enum):
    DOCUMENTARY = "documentary"
    REPRESENTATIVE = "representative"
    ABSTRACT = "abstract"


@dataclass(frozen=True)
class IdentityConstraint:
    """Truth constraint for identities represented in generated or edited assets."""

    entity_id: str
    preserve_identity: bool = True
    preserve_appearance: bool = True
    preserve_context: bool = True
    source_evidence_required: bool = True

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must not be blank.")


@dataclass(frozen=True)
class PromiseAlignment:
    aligned: bool
    represented_claims: tuple[str, ...] = ()
    supported_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    confidence: float = 0.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True)
class ThumbnailEvaluation:
    """Structured evaluation of one rendered thumbnail attempt."""

    accepted: bool
    technical_score: float = 0.0
    composition_score: float = 0.0
    readability_score: float = 0.0
    concept_fit_score: float = 0.0
    curiosity_score: float = 0.0
    visual_quality_score: float = 0.0
    truthfulness_score: float = 0.0
    promise_alignment: PromiseAlignment = PromiseAlignment(aligned=False)
    hard_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rationale: str = ""
    # Legacy aggregate score retained only during migration.
    score: float | None = None
    soft_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1 when provided.")

        for name, value in (
            ("technical_score", self.technical_score),
            ("composition_score", self.composition_score),
            ("readability_score", self.readability_score),
            ("concept_fit_score", self.concept_fit_score),
            ("curiosity_score", self.curiosity_score),
            ("visual_quality_score", self.visual_quality_score),
            ("truthfulness_score", self.truthfulness_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
