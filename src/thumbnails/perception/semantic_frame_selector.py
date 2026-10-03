from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from src.thumbnails.perception.frame_candidate_scorer import FrameCandidateScorer
from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.frame_selector import SelectedFrame
from src.thumbnails.perception.visual_intelligence import VisualIntelligenceResult


class SemanticFrameSelector:
    """
    Re-ranks ordinary frame candidates using semantic framing evidence.

    The base frame scorer remains responsible for image quality, subject
    prominence, focal evidence, and crop suitability. This selector adds one
    narrow concern: whether the important upper portion of the primary subject
    has enough breathing room from the source-frame edges.

    It is deliberately conservative. Semantic evidence can change which
    frame wins, but it cannot invent a frame or alter the base score contract.
    """

    def __init__(
        self,
        *,
        scorer: FrameCandidateScorer | None = None,
        semantic_weight: float = 0.35,
        preferred_edge_clearance: float = 0.08,
        minimum_text_space: float = 0.24,
    ) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1.")
        if preferred_edge_clearance <= 0.0:
            raise ValueError("preferred_edge_clearance must be positive.")
        if minimum_text_space <= 0.0 or minimum_text_space >= 0.5:
            raise ValueError("minimum_text_space must be between 0 and 0.5.")

        self.scorer = scorer or FrameCandidateScorer()
        self.semantic_weight = semantic_weight
        self.preferred_edge_clearance = preferred_edge_clearance
        self.minimum_text_space = minimum_text_space

    def select(
        self,
        candidates: Sequence[FramePerception],
        semantic_results: Mapping[Path, VisualIntelligenceResult],
    ) -> SelectedFrame:
        if not candidates:
            raise ValueError("candidates must not be empty.")

        ranked: list[tuple[float, FramePerception, object]] = []

        for perception in candidates:
            base_score = self.scorer.score(perception)
            semantic = semantic_results.get(perception.frame_path)
            semantic_score = self._score_semantic_framing(semantic)

            combined = (
                (1.0 - self.semantic_weight) * base_score.overall_score
                + self.semantic_weight * semantic_score
            )

            ranked.append((combined, perception, base_score))

        _, selected_perception, selected_score = max(
            ranked,
            key=lambda item: item[0],
        )

        return SelectedFrame(
            perception=selected_perception,
            score=selected_score,
        )

    def _score_semantic_framing(
        self,
        result: VisualIntelligenceResult | None,
    ) -> float:
        if result is None or result.primary_subject is None:
            return 0.0

        subject = result.primary_subject
        bounds = subject.bounds

        # The head is the visually sensitive upper portion of a person.
        # Keep it away from the source-frame edges because a vertical cover
        # cannot recover pixels that were already touching the source edge.
        head_bottom = bounds.top + bounds.height * 0.30
        head = (
            max(0.0, bounds.left),
            max(0.0, bounds.top),
            min(1.0, bounds.right),
            min(1.0, head_bottom),
        )

        left_clearance = head[0]
        top_clearance = head[1]
        right_clearance = 1.0 - head[2]

        def normalized(clearance: float) -> float:
            return min(1.0, max(0.0, clearance / self.preferred_edge_clearance))

        head_edge_score = min(
            normalized(left_clearance),
            normalized(top_clearance),
            normalized(right_clearance),
        )

        # A full-body edge contact is less serious than a head edge contact,
        # but excessive truncation still makes the source frame less useful.
        body_edge_clearance = min(
            bounds.left,
            bounds.top,
            1.0 - bounds.right,
            1.0 - bounds.bottom,
        )
        body_edge_score = normalized(body_edge_clearance)

        # Reward genuine side space strongly enough to overcome small,
        # reasonable differences in raw image quality. This is the evidence
        # the vertical composition needs to place text beside the subject.
        side_space = max(bounds.left, 1.0 - bounds.right)
        upper_space = max(0.0, bounds.top)
        lower_space = max(0.0, 1.0 - bounds.bottom)

        side_score = min(
            1.0,
            side_space / (self.minimum_text_space * 1.5),
        )
        vertical_score = min(
            1.0,
            max(upper_space, lower_space) / self.minimum_text_space,
        )

        return (
            0.45 * head_edge_score
            + 0.15 * body_edge_score
            + 0.35 * side_score
            + 0.05 * vertical_score
        )
