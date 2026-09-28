from pathlib import Path

import pytest

from src.config import ProjectWorkspace
from src.editing.planning.editing_planner import EditingPlanner
from src.editing.rendering.editing_renderer import (
    EditingRenderingError,
)
from src.intelligence.base import BaseAIDirector
from src.packaging.base import BasePackager
from src.pipeline.artifact_state import ArtifactStateChecker
from src.pipeline.orchestrator import PipelineOrchestrator
from src.rendering.ffmpeg_renderer import RenderingError
from src.rendering.source_provider import SourceAcquisitionError
from src.schemas import (
    CandidateEvaluation,
    CandidateManifest,
    CandidateWindow,
    ClipDecision,
    ClipManifest,
    EvaluationManifest,
    Transcript,
    TranscriptSegment,
    VideoSource,
)


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
    def __init__(self):
        self.calls = 0

    def package_clip(
        self,
        clip,
        transcript_text,
        source_title,
    ):
        self.calls += 1

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


class FakeSourceProvider:
    def __init__(self):
        self.acquire_calls = 0
        self.validate_calls = 0
        self.acquired_ranges = []

    def acquire_section(
        self,
        source_path,
        output_path,
        start_time,
        end_time,
    ):
        self.acquire_calls += 1
        self.acquired_ranges.append(
            (start_time, end_time)
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            "fake source section",
            encoding="utf-8",
        )

        return output_path

    def validate_media(self, media_path):
        self.validate_calls += 1

        if not media_path.exists():
            raise SourceAcquisitionError(
                f"Missing fake source section: {media_path}"
            )


class FakeBaseRenderer:
    def __init__(self):
        self.render_calls = 0
        self.validate_calls = 0

    def render_clip(
        self,
        source_path,
        output_path,
        start_time,
        end_time,
    ):
        self.render_calls += 1

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            "fake base render",
            encoding="utf-8",
        )

        assert start_time == 0.0
        assert end_time == 40.0

        return output_path

    def validate_media(self, media_path):
        self.validate_calls += 1

        if not media_path.exists():
            raise RenderingError(
                f"Missing fake base render: {media_path}"
            )


class FakeEditingRenderer:
    def __init__(self):
        self.render_calls = 0
        self.validate_calls = 0
        self.plans = []

    def render(
        self,
        source_path,
        output_path,
        plan,
    ):
        self.render_calls += 1
        self.plans.append(plan)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            "fake final edited clip",
            encoding="utf-8",
        )

        assert source_path.exists()
        assert plan.clip_id == "clip_01"

        return output_path

    def validate_media(self, media_path):
        self.validate_calls += 1

        if not media_path.exists():
            raise EditingRenderingError(
                f"Missing fake edited clip: {media_path}"
            )


def make_workspace(tmp_path):
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

    return workspace


def make_source():
    return VideoSource(
        source_type="local",
        location="test_source.mp4",
        title="Orchestrator Test",
    )


def make_candidate_manifest(source):
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

    return CandidateManifest(
        source=source,
        total_candidates=1,
        candidates=[candidate],
    )


def make_decision():
    return ClipDecision(
        clip_id=1,
        candidate_id=1,
        snapped_start_time=10.0,
        snapped_end_time=50.0,
        final_score=9.0,
        reason="Test clip.",
        title="Test Clip",
    )


def make_transcript(source):
    return Transcript(
        source=source,
        duration=60.0,
        language="en",
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


def test_orchestrator_packaging_checkpoint(tmp_path):
    workspace = make_workspace(tmp_path)
    source = make_source()
    candidate_manifest = make_candidate_manifest(source)
    decision = make_decision()

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

    second_packaging = orchestrator._ensure_packaging(
        decision=decision,
        candidate_manifest=candidate_manifest,
        source=source,
        workspace=workspace,
    )

    assert second_packaging.to_dict() == (
        packaging.to_dict()
    )


def test_orchestrator_runs_render_edit_and_packaging_checkpoints(
    tmp_path,
):
    workspace = make_workspace(tmp_path)

    source = make_source()
    candidate_manifest = make_candidate_manifest(
        source
    )
    decision = make_decision()

    clip_manifest = ClipManifest(
        source=source,
        selected_clips=[decision],
    )

    transcript = make_transcript(source)

    source_provider = FakeSourceProvider()
    base_renderer = FakeBaseRenderer()
    editing_renderer = FakeEditingRenderer()
    packager = FakePackager()

    orchestrator = PipelineOrchestrator(
        intelligence_engine=FakeAIDirector(),
        renderer=base_renderer,
        local_source_provider=source_provider,
        youtube_source_provider=source_provider,
        packager=packager,
        editing_planner=EditingPlanner(),
        editing_renderer=editing_renderer,
    )

    first_result = (
        orchestrator._ensure_rendered_and_packaged_clips(
            source=source,
            candidate_manifest=candidate_manifest,
            transcript=transcript,
            clip_manifest=clip_manifest,
            workspace=workspace,
        )
    )

    assert len(first_result) == 1

    final_path = Path(
        first_result[0].file_path
    )

    source_section_path = (
        workspace.source_dir
        / "section_01.mp4"
    )

    base_render_path = (
        workspace.renders_dir
        / "clip_01.mp4"
    )

    assert source_section_path.exists()
    assert base_render_path.exists()
    assert final_path.exists()

    assert source_provider.acquire_calls == 1
    assert source_provider.acquired_ranges == [
        (10.0, 50.0)
    ]

    assert base_renderer.render_calls == 1
    assert editing_renderer.render_calls == 1
    assert packager.calls == 1

    assert len(editing_renderer.plans) == 1

    first_plan = editing_renderer.plans[0]

    assert first_plan.clip_id == "clip_01"

    assert (
        first_plan.composition.canvas.width
        == 1080
    )

    assert (
        first_plan.composition.canvas.height
        == 1920
    )

    assert first_plan.captions.enabled is True
    assert first_plan.captions.segments

    first_caption = (
        first_plan.captions.segments[0]
    )

    assert first_caption.start_time == 0.0
    assert first_caption.end_time > 0.0

    second_result = (
        orchestrator._ensure_rendered_and_packaged_clips(
            source=source,
            candidate_manifest=candidate_manifest,
            transcript=transcript,
            clip_manifest=clip_manifest,
            workspace=workspace,
        )
    )

    assert len(second_result) == 1

    second_final_path = Path(
        second_result[0].file_path
    )

    assert second_final_path == final_path
    assert second_final_path.exists()

    assert source_provider.acquire_calls == 1
    assert base_renderer.render_calls == 1
    assert editing_renderer.render_calls == 1
    assert packager.calls == 1


@pytest.mark.parametrize(
    "title, expected_filename",
    [
        (
            "Test Clip",
            "clip_01_Test Clip.mp4",
        ),
        (
            "A: Test / Clip?",
            "clip_01_A_ Test _ Clip.mp4",
        ),
        (
            "   ---   ",
            "clip_01_untitled.mp4",
        ),
    ],
)
def test_orchestrator_uses_safe_final_clip_paths(
    tmp_path,
    title,
    expected_filename,
):
    workspace = make_workspace(tmp_path)

    checker = ArtifactStateChecker(
        workspace
    )

    actual_path = checker.final_clip_path(
        clip_id=1,
        title=title,
    )

    assert actual_path == (
        workspace.clips_dir
        / expected_filename
    )