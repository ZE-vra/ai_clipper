from __future__ import annotations

from dataclasses import dataclass

from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.frame_candidate_scorer import FrameCandidateScorer
from src.thumbnails.perception.scoring import FrameCandidateScore


@dataclass(frozen=True)
class SelectedFrame:
    perception: FramePerception
    score: FrameCandidateScore


class FrameSelector:
    """
    Selects the strongest frame from a set of perceived candidates.

    This component does not inspect images or make creative decisions.
    It ranks candidates using the configured FrameCandidateScorer.
    """

    def __init__(self, scorer: FrameCandidateScorer | None = None) -> None:
        self.scorer = scorer or FrameCandidateScorer()

    def select(self, candidates: list[FramePerception]) -> SelectedFrame:
        if not candidates:
            raise ValueError("candidates must not be empty.")

        scored_candidates = [
            (candidate, self.scorer.score(candidate))
            for candidate in candidates
        ]

        selected_candidate, selected_score = max(
            scored_candidates,
            key=lambda item: item[1].overall_score,
        )

        return SelectedFrame(
            perception=selected_candidate,
            score=selected_score,
        )