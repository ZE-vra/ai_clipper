from src.planning.clip_planner import ClipPlanner
from src.schemas import (
    CandidateEvaluation,
    CandidateManifest,
    CandidateWindow,
    EvaluationManifest,
    TranscriptSegment,
    VideoSource,
)


def _manifests(start_segment_id=None, end_segment_id=None):
    source = VideoSource(
        source_type="local",
        location="example.mp4",
    )

    segments = [
        TranscriptSegment(
            id=1,
            start=100.0,
            end=120.0,
            text="Here is the setup.",
        ),
        TranscriptSegment(
            id=2,
            start=120.0,
            end=145.0,
            text="Here is the key explanation.",
        ),
        TranscriptSegment(
            id=3,
            start=145.0,
            end=188.0,
            text="And here is the payoff.",
        ),
        TranscriptSegment(
            id=4,
            start=188.0,
            end=190.0,
            text="Unnecessary post-payoff material.",
        ),
    ]

    candidate = CandidateWindow(
        candidate_id=1,
        start_time=100.0,
        end_time=190.0,
        transcript_text=" ".join(segment.text for segment in segments),
        segments=segments,
    )

    candidate_manifest = CandidateManifest(
        source=source,
        total_candidates=1,
        candidates=[candidate],
    )

    evaluation_manifest = EvaluationManifest(
        source=source,
        evaluations=[
            CandidateEvaluation(
                candidate_id=1,
                score=9.2,
                reason="Strong payoff.",
                suggested_title="The Key Reveal",
                start_segment_id=start_segment_id,
                end_segment_id=end_segment_id,
            )
        ],
    )

    return candidate_manifest, evaluation_manifest


def test_planner_uses_ai_refined_boundaries():
    candidate_manifest, evaluation_manifest = _manifests(
        start_segment_id=1,
        end_segment_id=3,
    )

    result = ClipPlanner().plan(
        candidate_manifest,
        evaluation_manifest,
    )

    decision = result.selected_clips[0]

    assert decision.snapped_start_time == 100.0
    assert decision.snapped_end_time == 188.0
    assert decision.duration == 88.0


def test_planner_keeps_legacy_candidate_boundaries_when_refinement_missing():
    candidate_manifest, evaluation_manifest = _manifests()

    result = ClipPlanner().plan(
        candidate_manifest,
        evaluation_manifest,
    )

    decision = result.selected_clips[0]

    assert decision.snapped_start_time == 100.0
    assert decision.snapped_end_time == 190.0
    assert decision.duration == 90.0
