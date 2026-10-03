from pathlib import Path

from PIL import Image

from src.config import ProjectWorkspace
from src.packaging.models import ClipPackaging
from src.pipeline.orchestrator import PipelineOrchestrator
from src.schemas import ClipDecision, VideoSource


class MustNotRunThumbnailStage:
    def generate(self, **kwargs):
        raise AssertionError("valid thumbnail checkpoint should be reused")


def _workspace(root: Path) -> ProjectWorkspace:
    return ProjectWorkspace(
        project_id="test-project",
        root_dir=root,
        source_dir=root / "source",
        audio_dir=root / "audio",
        transcript_dir=root / "transcript",
        candidates_dir=root / "candidates",
        evaluations_dir=root / "evaluations",
        clips_dir=root / "clips",
        packaging_dir=root / "packaging",
        logs_dir=root / "logs",
    )


def test_thumbnail_checkpoint_skips_stage_and_avoids_any_ai_call(tmp_path: Path):
    workspace = _workspace(tmp_path)
    output = workspace.root_dir / "thumbnails" / "clip_01_thumbnail.jpg"
    output.parent.mkdir(parents=True)
    Image.new("RGB", (1080, 1920), (20, 30, 40)).save(output)

    pipeline = object.__new__(PipelineOrchestrator)
    pipeline.thumbnail_stage = MustNotRunThumbnailStage()

    pipeline._ensure_thumbnail(
        source=VideoSource(source_type="local", location=str(tmp_path / "source.mp4")),
        decision=ClipDecision(
            clip_id=1,
            candidate_id=1,
            snapped_start_time=0.0,
            snapped_end_time=10.0,
            final_score=9.0,
            reason="test",
            title="test",
        ),
        packaging=ClipPackaging(
            clip_id=1,
            title="Title",
            hook="A hook",
            caption="Caption",
            description="Description",
            thumbnail_text="Short hook",
            content_angle="Content angle",
            hashtags=[],
        ),
        workspace=workspace,
    )

    assert output.is_file()



class PortraitThumbnailStage:
    def __init__(self):
        self.called = False

    def generate(self, *, output_path, **kwargs):
        self.called = True
        Image.new("RGB", (1080, 1920), (40, 50, 60)).save(output_path)
        from src.thumbnails.domain.results import (
            ThumbnailResult,
            ThumbnailResultStatus,
        )
        return ThumbnailResult(
            status=ThumbnailResultStatus.SUCCESS,
            output_path=str(output_path),
        )


def test_landscape_thumbnail_checkpoint_is_regenerated_for_shorts(tmp_path: Path):
    workspace = _workspace(tmp_path)
    output = workspace.root_dir / "thumbnails" / "clip_01_thumbnail.jpg"
    output.parent.mkdir(parents=True)
    Image.new("RGB", (1280, 720), (20, 30, 40)).save(output)

    stage = PortraitThumbnailStage()
    pipeline = object.__new__(PipelineOrchestrator)
    pipeline.thumbnail_stage = stage
    pipeline._ensure_source_section = lambda **kwargs: tmp_path / "section.mp4"

    pipeline._ensure_thumbnail(
        source=VideoSource(source_type="local", location=str(tmp_path / "source.mp4")),
        decision=ClipDecision(
            clip_id=1,
            candidate_id=1,
            snapped_start_time=0.0,
            snapped_end_time=10.0,
            final_score=9.0,
            reason="test",
            title="test",
        ),
        packaging=ClipPackaging(
            clip_id=1,
            title="Title",
            hook="A hook",
            caption="Caption",
            description="Description",
            thumbnail_text="Short hook",
            content_angle="Content angle",
            hashtags=[],
        ),
        workspace=workspace,
    )

    assert stage.called
    with Image.open(output) as image:
        assert image.size == (1080, 1920)
