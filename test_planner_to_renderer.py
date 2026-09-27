from src.planning.clip_planner import ClipPlanner
from src.schemas import (
    CandidateEvaluation,
    CandidateManifest,
    CandidateWindow,
    EvaluationManifest,
    TranscriptSegment,
    VideoSource,
)


def test_planner_creates_clip_decision():
    source = VideoSource(
        source_type="local",
        location="test_source.mp4",
        title="Planner Test",
    )

    candidate = CandidateWindow(
        candidate_id=1,
        start_time=130.0,
        end_time=150.0,
        transcript_text="This is a strong test candidate.",
        segments=[
            TranscriptSegment(
                id=1,
                start=130.0,
                end=150.0,
                text="This is a strong test candidate.",
            )
        ],
    )

    candidate_manifest = CandidateManifest(
        source=source,
        total_candidates=1,
        candidates=[candidate],
    )

    evaluation = CandidateEvaluation(
        candidate_id=1,
        score=9.0,
        reason="Strong test candidate.",
        suggested_title="Planner Test Clip",
    )

    evaluation_manifest = EvaluationManifest(
        source=source,
        evaluations=[evaluation],
    )

    planner = ClipPlanner()

    clip_manifest = planner.plan(
        candidate_manifest=candidate_manifest,
        evaluation_manifest=evaluation_manifest,
    )

    assert len(clip_manifest.selected_clips) == 1

    decision = clip_manifest.selected_clips[0]

    assert decision.candidate_id == 1
    assert decision.final_score == 9.0
    assert decision.title == "Planner Test Clip"
    assert decision.snapped_end_time > decision.snapped_start_time
    assert decision.duration > 0