"""Module 5: Clip Planning.

Transforms AI candidate evaluations into deterministic clip decisions.

The AI Director decides which moments are compelling.
The planner decides which of those moments become final clips.
"""

import json
from pathlib import Path
from typing import List, Optional

from src.schemas import (
    CandidateManifest,
    ClipDecision,
    ClipManifest,
    EvaluationManifest,
)


class ClipPlanner:
    """Deterministically selects final clips from AI evaluations."""

    def __init__(
        self,
        min_score: float = 7.0,
        max_clips: int = 3,
        min_duration: float = 15.0,
        max_duration: float = 180.0,
        overlap_threshold: float = 0.5,
    ):
        self.min_score = min_score
        self.max_clips = max_clips
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.overlap_threshold = overlap_threshold

    def plan(
        self,
        candidate_manifest: CandidateManifest,
        evaluation_manifest: EvaluationManifest,
    ) -> ClipManifest:
        """Select the strongest non-overlapping candidates."""

        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in candidate_manifest.candidates
        }

        eligible = [
            evaluation
            for evaluation in evaluation_manifest.evaluations
            if evaluation.score >= self.min_score
        ]

        eligible.sort(
            key=lambda evaluation: evaluation.score,
            reverse=True,
        )

        selected: List[ClipDecision] = []

        for evaluation in eligible:
            candidate = candidates_by_id.get(
                evaluation.candidate_id
            )

            if candidate is None:
                continue

            start, end = self._resolve_boundaries(
                candidate,
                evaluation,
            )
            duration = end - start

            if duration < self.min_duration:
                continue

            if duration > self.max_duration:
                continue

            if self._overlaps_existing(
                start,
                end,
                selected,
            ):
                continue

            clip_id = len(selected) + 1

            decision = ClipDecision(
                clip_id=clip_id,
                candidate_id=candidate.candidate_id,
                snapped_start_time=round(start, 3),
                snapped_end_time=round(end, 3),
                final_score=evaluation.score,
                reason=evaluation.reason,
                title=evaluation.suggested_title,
            )

            selected.append(decision)

            if len(selected) >= self.max_clips:
                break

        return ClipManifest(
            source=candidate_manifest.source,
            selected_clips=selected,
        )

    @staticmethod
    def _resolve_boundaries(
        candidate,
        evaluation,
    ) -> tuple[float, float]:
        """Use Gemini's segment boundaries when available and valid."""
        if (
            evaluation.start_segment_id is None
            or evaluation.end_segment_id is None
        ):
            return candidate.start_time, candidate.end_time

        segments_by_id = {
            segment.id: segment
            for segment in candidate.segments
        }

        start_segment = segments_by_id.get(
            evaluation.start_segment_id
        )
        end_segment = segments_by_id.get(
            evaluation.end_segment_id
        )

        if start_segment is None or end_segment is None:
            return candidate.start_time, candidate.end_time

        if end_segment.end <= start_segment.start:
            return candidate.start_time, candidate.end_time

        return (
            start_segment.start,
            end_segment.end,
        )

    def _overlaps_existing(
        self,
        start: float,
        end: float,
        selected: List[ClipDecision],
    ) -> bool:
        """Check whether a candidate overlaps an existing clip too much."""

        for existing in selected:
            overlap = self._overlap_ratio(
                start,
                end,
                existing.snapped_start_time,
                existing.snapped_end_time,
            )

            if overlap >= self.overlap_threshold:
                return True

        return False

    @staticmethod
    def _overlap_ratio(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        """Calculate overlap relative to the shorter clip."""

        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)

        if overlap_end <= overlap_start:
            return 0.0

        overlap = overlap_end - overlap_start

        duration_a = end_a - start_a
        duration_b = end_b - start_b

        shortest_duration = min(
            duration_a,
            duration_b,
        )

        if shortest_duration <= 0:
            return 0.0

        return overlap / shortest_duration