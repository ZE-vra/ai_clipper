from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThumbnailEvaluation:
    """Evaluation produced after a thumbnail attempt."""

    accepted: bool

    hard_failures: tuple[str, ...] = ()
    soft_failures: tuple[str, ...] = ()

    score: Optional[float] = None

    reason: str = ""
