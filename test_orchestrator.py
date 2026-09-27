from src.pipeline.orchestrator import PipelineOrchestrator
from src.packaging.base import BasePackager
from src.intelligence.base import BaseAIDirector
from src.schemas import (
    CandidateEvaluation,
    CandidateManifest,
    CandidateWindow,
    ClipDecision,
    EvaluationManifest,
    TranscriptSegment,
    VideoSource,
)
from src.config import ProjectWorkspace


class FakeAIDirector(BaseAIDirector):
    def evaluate_candidates(self, candidate_manifest):
        return EvaluationManifest(
            source=candidate_manifest.source,
            evaluations=[
                CandidateEvaluation(
                    candidate_id=1,
                    score=9.0,
                    reason="Test evaluation.",
                    suggested_title="Test Clip",
                )
            ],
        )


class FakePackager(BasePackager):
    def package_clip(
        self,
        clip,
        transcript_text,
        source_title,
    ):
        assert transcript_text == (
            "This is the transcript belonging "
            "to the selected candidate."
        )

        assert source_title == "Orchestrator Test"

        return {
            "title": "Test Packaging Title",
            "hook": "Test packaging hook.",
            "caption": "Test packaging caption.",
            "description": "Test packaging description.",
            "thumbnail_text": "TEST PACKAGING",
            "content_angle": "test",
            "hashtags": [
                "#test",
                "#packaging",
            ],
        }


def test_orchestrator_packaging_checkpoint(tmp_path):
    workspace = ProjectWorkspace(
        project_id="orchestrator_test",
        root_dir=tmp_path,
        source_dir=tmp_path / "source",
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcript",
        candidates_dir=tmp_path / "candidates",
        evaluations_dir=tmp_path / "evaluations",
        clips_dir=tmp_path / "clips",
        packaging_dir=tmp_path / "packaging",
        logs_dir=tmp_path / "logs",
    )

    workspace.initialize()

    source = VideoSource(
        source_type="local",
        location="test_source.mp4",
        title="Orchestrator Test",
    )

    candidate = CandidateWindow(
        candidate_id=1,
        start_time=10.0,
        end_time=50.0,
        transcript_text=(
            "This is the transcript belonging "
            "to the selected candidate."
        ),
        segments=[
            TranscriptSegment(
                id=1,
                start=10.0,
                end=50.0,
                text=(
                    "This is the transcript belonging "
                    "to the selected candidate."
                ),
            )
        ],
    )

    candidate_manifest = CandidateManifest(
        source=source,
        total_candidates=1,
        candidates=[candidate],
    )

    decision = ClipDecision(
        clip_id=1,
        candidate_id=1,
        snapped_start_time=10.0,
        snapped_end_time=50.0,
        final_score=9.0,
        reason="Test clip.",
        title="Test Clip",
    )

    orchestrator = PipelineOrchestrator(
        intelligence_engine=FakeAIDirector(),
        packager=FakePackager(),
    )

    packaging = orchestrator._ensure_packaging(
        decision=decision,
        candidate_manifest=candidate_manifest,
        source=source,
        workspace=workspace,
    )

    assert packaging.clip_id == 1
    assert packaging.title == "Test Packaging Title"
    assert packaging.hook == "Test packaging hook."
    assert packaging.hashtags == [
        "#test",
        "#packaging",
    ]

    packaging_path = (
        workspace.packaging_dir / "clip_01.json"
    )

    assert packaging_path.exists()

    # Running the checkpoint again should reuse the saved artifact.
    second_packaging = orchestrator._ensure_packaging(
        decision=decision,
        candidate_manifest=candidate_manifest,
        source=source,
        workspace=workspace,
    )

    assert second_packaging.to_dict() == (
        packaging.to_dict()
    )