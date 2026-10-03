from pathlib import Path

from PIL import Image

from src.config import Config
from src.packaging.models import ClipPackaging
from src.packaging.persistence import save_clip_packaging
from src.thumbnails.domain.results import ThumbnailResult, ThumbnailResultStatus
from src.thumbnails import project_cli


class FakeThumbnailStage:
    calls = []

    def __init__(self, *, sample_count=9):
        self.sample_count = sample_count

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        Image.new("RGB", (1080, 1920), (30, 40, 50)).save(output_path)
        return ThumbnailResult(
            status=ThumbnailResultStatus.SUCCESS,
            output_path=str(output_path),
        )


def test_project_cli_regenerates_only_one_thumbnail_from_checkpoints(
    tmp_path: Path, monkeypatch
):
    workspace = Config.workspace_from_project_id("test-project")
    # Redirect the project workspace to an isolated test directory.
    workspace.root_dir = tmp_path
    workspace.source_dir = tmp_path / "source"
    workspace.packaging_dir = tmp_path / "packaging"
    workspace.source_dir.mkdir()
    workspace.packaging_dir.mkdir()
    section = workspace.source_dir / "section_02.mp4"
    section.write_bytes(b"existing source section")
    save_clip_packaging(
        ClipPackaging(
            clip_id=2,
            title="Title",
            hook="Hook",
            caption="Caption",
            description="Description",
            thumbnail_text="Short hook",
            content_angle="A clear content angle",
            hashtags=[],
        ),
        workspace,
    )

    FakeThumbnailStage.calls = []
    monkeypatch.setattr(project_cli.Config, "find_workspace", lambda source: workspace)
    monkeypatch.setattr(project_cli, "ThumbnailV2Stage", FakeThumbnailStage)

    result = project_cli.main(["existing-source.mp4", "--clip-id", "2", "--samples", "5"])

    assert result == 0
    assert len(FakeThumbnailStage.calls) == 1
    call = FakeThumbnailStage.calls[0]
    assert call["source_video_path"] == section
    assert call["packaging"].clip_id == 2
    assert call["candidates_dir"] == tmp_path / "thumbnail_work" / "clip_02"
    assert call["output_path"] == tmp_path / "thumbnails" / "clip_02_thumbnail.jpg"
    with Image.open(call["output_path"]) as image:
        assert image.size == (1080, 1920)


def test_project_cli_fails_without_saved_source_section(tmp_path: Path, monkeypatch):
    workspace = Config.workspace_from_project_id("test-project")
    workspace.root_dir = tmp_path
    workspace.source_dir = tmp_path / "source"
    workspace.packaging_dir = tmp_path / "packaging"
    workspace.source_dir.mkdir()
    workspace.packaging_dir.mkdir()
    save_clip_packaging(
        ClipPackaging(
            clip_id=1,
            title="Title",
            hook="Hook",
            caption="Caption",
            description="Description",
            thumbnail_text="Short hook",
            content_angle="Angle",
            hashtags=[],
        ),
        workspace,
    )
    monkeypatch.setattr(project_cli.Config, "find_workspace", lambda source: workspace)

    result = project_cli.main(["existing-source.mp4", "--clip-id", "1"])

    assert result == 1
