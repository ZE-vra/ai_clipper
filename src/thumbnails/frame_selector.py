from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.thumbnails.models import ThumbnailFrameCandidate


class ThumbnailFrameSelector(Protocol):
    """
    Selects the source frame to use for a thumbnail.

    Candidate generation and final selection are intentionally exposed
    separately so a future visual/AI scorer can replace the deterministic
    strategy without changing the rest of the thumbnail pipeline.
    """

    def generate_candidates(
        self,
        duration: float,
    ) -> list[ThumbnailFrameCandidate]:
        ...

    def select(
        self,
        candidates: Sequence[ThumbnailFrameCandidate],
    ) -> ThumbnailFrameCandidate:
        ...


@dataclass(frozen=True)
class DeterministicFrameSelector:
    """
    Baseline frame-selection strategy.

    Candidate timestamps are distributed across the clip while avoiding
    the exact start and end frames, which are more likely to contain
    transitions, fades, or incomplete visual states.

    The current selector chooses the middle candidate.

    This is intentionally a replaceable baseline, not a claim that the
    middle frame is semantically optimal.
    """

    candidate_positions: tuple[float, ...] = (
        0.10,
        0.25,
        0.40,
        0.55,
        0.70,
        0.85,
    )

    def __post_init__(self) -> None:
        if not self.candidate_positions:
            raise ValueError(
                "candidate_positions must not be empty."
            )

        previous = -1.0

        for position in self.candidate_positions:
            if not 0.0 < position < 1.0:
                raise ValueError(
                    "candidate positions must be between 0.0 and 1.0."
                )

            if position <= previous:
                raise ValueError(
                    "candidate positions must be strictly increasing."
                )

            previous = position

    def generate_candidates(
        self,
        duration: float,
    ) -> list[ThumbnailFrameCandidate]:
        if duration <= 0:
            raise ValueError(
                "duration must be greater than zero."
            )

        return [
            ThumbnailFrameCandidate(
                timestamp=duration * position
            )
            for position in self.candidate_positions
        ]

    def select(
        self,
        candidates: Sequence[ThumbnailFrameCandidate],
    ) -> ThumbnailFrameCandidate:
        if not candidates:
            raise ValueError(
                "candidates must not be empty."
            )

        return candidates[len(candidates) // 2]