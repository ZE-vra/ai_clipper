from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.thumbnails.domain.concepts import ThumbnailConcept
from src.thumbnails.domain.evaluation import ThumbnailEvaluation
from src.thumbnails.domain.plans import ThumbnailRenderPlan


class AttemptStatus(str, Enum):
    PLANNED = "planned"
    RENDERED = "rendered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class ThumbnailAttempt:
    """Immutable record of one complete thumbnail attempt."""

    attempt_id: str
    concept: ThumbnailConcept
    render_plan: ThumbnailRenderPlan
    status: AttemptStatus

    evaluation: Optional[ThumbnailEvaluation] = None

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError(
                "attempt_id must not be blank."
            )

        if (
            self.status
            in {
                AttemptStatus.ACCEPTED,
                AttemptStatus.REJECTED,
            }
            and self.evaluation is None
        ):
            raise ValueError(
                "accepted or rejected attempts must have "
                "an evaluation."
            )
