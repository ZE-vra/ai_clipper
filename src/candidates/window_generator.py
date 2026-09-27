"""Module 3: Candidate Discovery

Generates distinct narrative candidate windows from transcript segments.

The candidate generator proposes plausible sections of a video.
It does NOT decide whether a section is interesting.

The generator looks for natural narrative boundaries using transcript
pauses, creates natural-length windows, and advances substantially after
each accepted candidate to avoid producing many near-identical windows.

A controlled final-tail fallback ensures that substantial material near
the end of a video is not silently discarded simply because it lacks a
strong natural starting pause.
"""

from typing import List, Optional

from src.config import ProjectWorkspace
from src.exceptions import CandidateGenerationError
from src.persistence.manifests import save_candidate_manifest
from src.schemas import (
    CandidateManifest,
    CandidateWindow,
    Transcript,
)


def _pause_before_segment(
    segments,
    start_idx: int,
) -> float:
    """Return the silence/pause immediately before a segment."""

    if start_idx <= 0:
        return 999.0

    return max(
        0.0,
        segments[start_idx].start
        - segments[start_idx - 1].end,
    )


def _is_natural_start(
    segments,
    start_idx: int,
    pause_threshold: float,
) -> bool:
    """Determine whether a transcript segment is a good candidate start."""

    if start_idx == 0:
        return True

    pause = _pause_before_segment(
        segments,
        start_idx,
    )

    return pause >= pause_threshold


def _find_candidate_end(
    segments,
    start_idx: int,
    min_seconds: float,
    max_seconds: float,
    target_seconds: float,
    pause_threshold: float,
) -> Optional[int]:
    """Find the best natural ending for a candidate."""

    start_time = segments[start_idx].start

    minimum_end_idx = None

    for end_idx in range(
        start_idx,
        len(segments),
    ):
        duration = (
            segments[end_idx].end
            - start_time
        )

        if duration >= min_seconds:
            minimum_end_idx = end_idx
            break

    if minimum_end_idx is None:
        return None

    best_end_idx = None
    best_score = float("-inf")

    fallback_end_idx = None

    for end_idx in range(
        minimum_end_idx,
        len(segments),
    ):
        duration = (
            segments[end_idx].end
            - start_time
        )

        if duration > max_seconds:
            break

        fallback_end_idx = end_idx

        # The final transcript segment is automatically a possible
        # ending because there is no following pause to inspect.
        if end_idx == len(segments) - 1:
            distance_from_target = abs(
                duration - target_seconds
            )

            score = -distance_from_target

            if score > best_score:
                best_score = score
                best_end_idx = end_idx

            continue

        pause_after = (
            segments[end_idx + 1].start
            - segments[end_idx].end
        )

        if pause_after < pause_threshold:
            continue

        distance_from_target = abs(
            duration - target_seconds
        )

        pause_bonus = min(
            pause_after,
            3.0,
        ) * 3.0

        score = (
            -distance_from_target
            + pause_bonus
        )

        if score > best_score:
            best_score = score
            best_end_idx = end_idx

    if best_end_idx is not None:
        return best_end_idx

    return fallback_end_idx


def _build_candidate(
    segments,
    start_idx: int,
    end_idx: int,
    candidate_id: int,
) -> Optional[CandidateWindow]:
    """Build a CandidateWindow from transcript segments."""

    if end_idx < start_idx:
        return None

    window_segments = segments[
        start_idx : end_idx + 1
    ]

    if not window_segments:
        return None

    start_time = window_segments[0].start
    end_time = window_segments[-1].end

    transcript_text = " ".join(
        segment.text
        for segment in window_segments
    ).strip()

    if not transcript_text:
        return None

    return CandidateWindow(
        candidate_id=candidate_id,
        start_time=round(
            start_time,
            3,
        ),
        end_time=round(
            end_time,
            3,
        ),
        transcript_text=transcript_text,
        segments=list(window_segments),
    )


def _overlap_ratio(
    candidate_a: CandidateWindow,
    candidate_b: CandidateWindow,
) -> float:
    """Calculate overlap relative to the shorter candidate."""

    overlap_start = max(
        candidate_a.start_time,
        candidate_b.start_time,
    )

    overlap_end = min(
        candidate_a.end_time,
        candidate_b.end_time,
    )

    if overlap_end <= overlap_start:
        return 0.0

    overlap_duration = (
        overlap_end - overlap_start
    )

    shortest_duration = min(
        candidate_a.duration,
        candidate_b.duration,
    )

    if shortest_duration <= 0:
        return 0.0

    return overlap_duration / shortest_duration


def _is_near_duplicate(
    candidate: CandidateWindow,
    existing_candidates: List[CandidateWindow],
    overlap_threshold: float,
    boundary_tolerance: float,
) -> bool:
    """Reject candidates that are effectively the same narrative window."""

    for existing in existing_candidates:

        overlap = _overlap_ratio(
            candidate,
            existing,
        )

        if overlap >= overlap_threshold:
            return True

        start_difference = abs(
            candidate.start_time
            - existing.start_time
        )

        end_difference = abs(
            candidate.end_time
            - existing.end_time
        )

        if (
            start_difference <= boundary_tolerance
            and end_difference <= boundary_tolerance
        ):
            return True

    return False


def _find_next_natural_start(
    segments,
    current_start_idx: int,
    earliest_start_time: float,
    pause_threshold: float,
) -> Optional[int]:
    """Find the next natural starting point at or after a given time."""

    for idx in range(
        current_start_idx + 1,
        len(segments),
    ):
        if segments[idx].start < earliest_start_time:
            continue

        if _is_natural_start(
            segments,
            idx,
            pause_threshold=pause_threshold,
        ):
            return idx

    return None


def _advance_after_candidate(
    segments,
    start_idx: int,
    candidate: CandidateWindow,
    min_advance_seconds: float,
    min_progress_ratio: float,
    pause_threshold: float,
) -> Optional[int]:
    """Find the next natural start after meaningful progress.

    Progress is measured from the candidate's START, not its END.

    This prevents the generator from repeatedly starting only a few
    seconds later while still allowing later parts of the video,
    including the final section, to be discovered.
    """

    progress_required = max(
        min_advance_seconds,
        candidate.duration * min_progress_ratio,
    )

    earliest_start_time = (
        candidate.start_time
        + progress_required
    )

    return _find_next_natural_start(
        segments=segments,
        current_start_idx=start_idx,
        earliest_start_time=earliest_start_time,
        pause_threshold=pause_threshold,
    )


def _build_final_tail_candidate(
    segments,
    earliest_start_time: float,
    min_seconds: float,
    max_seconds: float,
    candidate_id: int,
) -> Optional[CandidateWindow]:
    """Build one final candidate covering substantial remaining material.

    This is intentionally a fallback, not a normal discovery mechanism.

    If normal natural-start discovery reaches the end of the transcript
    while a substantial amount of video remains uncovered, we allow one
    final candidate to begin at the first transcript segment after the
    fallback threshold.

    The AI Director will later decide whether this tail is actually
    interesting.
    """

    start_idx = None

    for idx, segment in enumerate(segments):
        if segment.start >= earliest_start_time:
            start_idx = idx
            break

    if start_idx is None:
        return None

    remaining_duration = (
        segments[-1].end
        - segments[start_idx].start
    )

    if remaining_duration < min_seconds:
        return None

    end_idx = len(segments) - 1

    # If the entire remaining tail is longer than the maximum candidate
    # duration, find the best ending within the allowed range.
    if remaining_duration > max_seconds:
        end_idx = None

        for idx in range(
            start_idx,
            len(segments),
        ):
            duration = (
                segments[idx].end
                - segments[start_idx].start
            )

            if duration > max_seconds:
                break

            end_idx = idx

        if end_idx is None:
            return None

    return _build_candidate(
        segments=segments,
        start_idx=start_idx,
        end_idx=end_idx,
        candidate_id=candidate_id,
    )


def generate_candidate_windows(
    transcript: Transcript,
    workspace: ProjectWorkspace,
    min_seconds: float = 30.0,
    max_seconds: float = 90.0,
    target_seconds: float = 55.0,
    pause_threshold: float = 0.8,
    overlap_threshold: float = 0.85,
    boundary_tolerance: float = 8.0,
    min_advance_seconds: float = 25.0,
    min_progress_ratio: float = 0.5,
    max_candidates: int = 40,
    final_tail_min_seconds: float = 45.0,
) -> CandidateManifest:
    """Generate diverse narrative candidate windows."""

    segments = transcript.segments

    if not segments:
        raise CandidateGenerationError(
            "Cannot generate candidates from an empty transcript."
        )

    if min_seconds <= 0:
        raise CandidateGenerationError(
            "min_seconds must be greater than 0."
        )

    if max_seconds <= min_seconds:
        raise CandidateGenerationError(
            "max_seconds must be greater than min_seconds."
        )

    if not min_seconds <= target_seconds <= max_seconds:
        raise CandidateGenerationError(
            "target_seconds must be between "
            "min_seconds and max_seconds."
        )

    if pause_threshold < 0:
        raise CandidateGenerationError(
            "pause_threshold cannot be negative."
        )

    if not 0 < overlap_threshold <= 1:
        raise CandidateGenerationError(
            "overlap_threshold must be greater than 0 "
            "and less than or equal to 1."
        )

    if boundary_tolerance < 0:
        raise CandidateGenerationError(
            "boundary_tolerance cannot be negative."
        )

    if min_advance_seconds <= 0:
        raise CandidateGenerationError(
            "min_advance_seconds must be greater than 0."
        )

    if not 0 < min_progress_ratio <= 1:
        raise CandidateGenerationError(
            "min_progress_ratio must be greater than 0 "
            "and less than or equal to 1."
        )

    if max_candidates <= 0:
        raise CandidateGenerationError(
            "max_candidates must be greater than 0."
        )

    if final_tail_min_seconds < min_seconds:
        raise CandidateGenerationError(
            "final_tail_min_seconds must be at least "
            "min_seconds."
        )

    candidates: List[CandidateWindow] = []

    candidate_id = 1
    start_idx = 0
    total_segments = len(segments)

    last_covered_end_time = segments[0].start

    while (
        start_idx < total_segments
        and len(candidates) < max_candidates
    ):

        # -----------------------------------------------------
        # FIND A NATURAL START
        # -----------------------------------------------------

        if not _is_natural_start(
            segments,
            start_idx,
            pause_threshold=pause_threshold,
        ):
            start_idx += 1
            continue

        # -----------------------------------------------------
        # FIND A NATURAL END
        # -----------------------------------------------------

        end_idx = _find_candidate_end(
            segments=segments,
            start_idx=start_idx,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            target_seconds=target_seconds,
            pause_threshold=pause_threshold,
        )

        if end_idx is None:
            break

        # -----------------------------------------------------
        # BUILD CANDIDATE
        # -----------------------------------------------------

        candidate = _build_candidate(
            segments=segments,
            start_idx=start_idx,
            end_idx=end_idx,
            candidate_id=candidate_id,
        )

        if candidate is None:
            start_idx += 1
            continue

        if candidate.duration < min_seconds:
            start_idx += 1
            continue

        if candidate.duration > max_seconds:
            start_idx += 1
            continue

        # -----------------------------------------------------
        # DUPLICATE CHECK
        # -----------------------------------------------------

        if not _is_near_duplicate(
            candidate=candidate,
            existing_candidates=candidates,
            overlap_threshold=overlap_threshold,
            boundary_tolerance=boundary_tolerance,
        ):
            candidates.append(candidate)
            candidate_id += 1

            last_covered_end_time = max(
                last_covered_end_time,
                candidate.end_time,
            )

        # -----------------------------------------------------
        # ADVANCE
        # -----------------------------------------------------

        next_start_idx = _advance_after_candidate(
            segments=segments,
            start_idx=start_idx,
            candidate=candidate,
            min_advance_seconds=min_advance_seconds,
            min_progress_ratio=min_progress_ratio,
            pause_threshold=pause_threshold,
        )

        # No later natural start exists.
        if next_start_idx is None:
            break

        # Always guarantee forward progress.
        if next_start_idx <= start_idx:
            next_start_idx = start_idx + 1

        start_idx = next_start_idx

    # ---------------------------------------------------------
    # FINAL TAIL FALLBACK
    # ---------------------------------------------------------

    if (
        len(candidates) < max_candidates
        and segments[-1].end - last_covered_end_time
        >= final_tail_min_seconds
    ):
        tail_earliest_start = max(
            last_covered_end_time,
            segments[0].start,
        )

        tail_candidate = _build_final_tail_candidate(
            segments=segments,
            earliest_start_time=tail_earliest_start,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            candidate_id=candidate_id,
        )

        if (
            tail_candidate is not None
            and not _is_near_duplicate(
                candidate=tail_candidate,
                existing_candidates=candidates,
                overlap_threshold=overlap_threshold,
                boundary_tolerance=boundary_tolerance,
            )
        ):
            candidates.append(tail_candidate)

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------

    if not candidates:
        full_text = " ".join(
            segment.text
            for segment in segments
        ).strip()

        if not full_text:
            raise CandidateGenerationError(
                "Transcript contains no usable text."
            )

        candidates.append(
            CandidateWindow(
                candidate_id=1,
                start_time=segments[0].start,
                end_time=segments[-1].end,
                transcript_text=full_text,
                segments=list(segments),
            )
        )

    manifest = CandidateManifest(
        source=transcript.source,
        total_candidates=len(candidates),
        candidates=candidates,
    )

    save_candidate_manifest(
        manifest,
        workspace,
    )

    return manifest