from pathlib import Path

from src.config import Config
from src.pipeline.orchestrator import PipelineOrchestrator
from src.rendering.ffmpeg_renderer import FFmpegRenderer
from src.schemas import ClipDecision, ClipManifest, VideoSource


SOURCE_PATH = Path(
    r"C:\Users\Admin\Videos\videos\b9cc7a86cb57a79164acc05df3a75f16_1790343864860.mp4"
)


class FailingLocalSourceProvider:
    """Test double that fails if source acquisition is attempted."""

    def acquire_section(
        self,
        source_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        raise AssertionError(
            "LocalSourceProvider was called even though the existing "
            "rendered clip was valid."
        )


class FailingRenderer(FFmpegRenderer):
    """Test renderer that fails if rendering is attempted."""

    def render_clip(
        self,
        source_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        raise AssertionError(
            "FFmpegRenderer.render_clip() was called even though the "
            "existing rendered clip was valid."
        )


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Test source does not exist: {SOURCE_PATH}"
        )

    workspace = Config.create_workspace(str(SOURCE_PATH))

    source = VideoSource(
        source_type="local",
        location=str(SOURCE_PATH.resolve()),
    )

    decision = ClipDecision(
        clip_id=1,
        candidate_id=1,
        snapped_start_time=100.0,
        snapped_end_time=110.0,
        final_score=10.0,
        reason="Valid rendered output resume test.",
        title="Valid Resume Skip Test",
    )

    clip_manifest = ClipManifest(
        source=source,
        selected_clips=[decision],
    )

    output_path = (
        workspace.clips_dir
        / "clip_01_Valid Resume Skip Test.mp4"
    )

    acquired_path = (
        workspace.source_dir
        / "section_01.mp4"
    )

    if output_path.exists():
        output_path.unlink()

    if acquired_path.exists():
        acquired_path.unlink()

    print("=" * 60)
    print("VALID RESUME SKIP TEST")
    print("=" * 60)
    print(f"Source:     {SOURCE_PATH}")
    print(f"Workspace:  {workspace.root_dir}")
    print("Range:      100.00s → 110.00s")
    print()

    # ---------------------------------------------------------
    # STEP 1
    # Create a real, valid final clip.
    # ---------------------------------------------------------

    print("=" * 60)
    print("STEP 1: CREATE VALID CLIP")
    print("=" * 60)

    setup_orchestrator = PipelineOrchestrator(
        intelligence_engine=object(),
    )

    setup_results = setup_orchestrator._ensure_rendered_clips(
        source=source,
        clip_manifest=clip_manifest,
        workspace=workspace,
    )

    if len(setup_results) != 1:
        raise AssertionError(
            f"Expected exactly 1 setup result, got {len(setup_results)}."
        )

    if not output_path.exists():
        raise AssertionError(
            "Expected the setup render to create the output."
        )

    original_size = output_path.stat().st_size

    print()
    print(f"Valid clip created: {output_path}")
    print(f"Size: {original_size:,} bytes")
    print()

    # ---------------------------------------------------------
    # STEP 2
    # Run the orchestrator again with providers that would
    # immediately fail if acquisition or rendering occurs.
    # ---------------------------------------------------------

    print("=" * 60)
    print("STEP 2: RESUME WITH VALID OUTPUT")
    print("=" * 60)

    resume_orchestrator = PipelineOrchestrator(
        intelligence_engine=object(),
        renderer=FailingRenderer(),
        local_source_provider=FailingLocalSourceProvider(),
    )

    results = resume_orchestrator._ensure_rendered_clips(
        source=source,
        clip_manifest=clip_manifest,
        workspace=workspace,
    )

    if len(results) != 1:
        raise AssertionError(
            f"Expected exactly 1 resumed result, got {len(results)}."
        )

    if not output_path.exists():
        raise AssertionError(
            "The valid existing output disappeared during resume."
        )

    final_size = output_path.stat().st_size

    if final_size != original_size:
        raise AssertionError(
            "The existing valid output was modified or replaced."
        )

    # ---------------------------------------------------------
    # STEP 3
    # Explicitly validate that the skipped file is still valid.
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 3: VERIFY EXISTING OUTPUT")
    print("=" * 60)

    FFmpegRenderer().validate_media(output_path)

    print()
    print("=" * 60)
    print("VALID RESUME SKIP SUCCEEDED")
    print("=" * 60)
    print(f"File: {output_path}")
    print(f"Size: {final_size:,} bytes")
    print()
    print(
        "Successfully verified:"
        "\n  Existing output"
        "\n      ↓"
        "\n  Validation passed"
        "\n      ↓"
        "\n  Acquisition skipped"
        "\n      ↓"
        "\n  Rendering skipped"
        "\n      ↓"
        "\n  Existing clip reused"
    )


if __name__ == "__main__":
    main()