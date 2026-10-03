from pathlib import Path

from src.packaging.models import ClipPackaging
from src.thumbnails.domain.results import (
    ThumbnailResult,
    ThumbnailResultStatus,
)
from src.thumbnails.pipeline_stage import ThumbnailV2Stage


class FakeOrchestrator:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return ThumbnailResult(
            status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
            failure_reason="test result",
        )


def test_v2_stage_uses_packaging_and_does_not_require_a_model_provider(tmp_path: Path):
    orchestrator = FakeOrchestrator()
    stage = ThumbnailV2Stage(orchestrator=orchestrator)
    packaging = ClipPackaging(
        clip_id=1,
        title="The Unexpected Result",
        hook="WHY IT CHANGED",
        caption="A concise caption.",
        description="A clip description.",
        thumbnail_text="THE RESULT CHANGED",
        content_angle="A surprising experiment result",
        hashtags=["#science"],
    )

    result = stage.generate(
        source_video_path=tmp_path / "section.mp4",
        packaging=packaging,
        output_path=tmp_path / "thumbnail.jpg",
        candidates_dir=tmp_path / "work",
    )

    assert result.status is ThumbnailResultStatus.NO_ACCEPTABLE_RESULT
    assert orchestrator.kwargs is not None
    assert orchestrator.kwargs["brief"].core_hook == "THE RESULT CHANGED"
    assert orchestrator.kwargs["brief"].promise == "The Unexpected Result"
    assert orchestrator.kwargs["understanding"].entities == ()
    assert orchestrator.kwargs["understanding"].events == ()
    assert orchestrator.kwargs["target"].size.width == 1280
    assert orchestrator.kwargs["target"].size.height == 720
