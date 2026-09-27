from pathlib import Path

from src.config import Config
from src.pipeline.artifact_state import ArtifactStateChecker
from src.pipeline.orchestrator import PipelineOrchestrator
from src.schemas import ClipDecision, ClipManifest, VideoSource


SOURCE_PATH = Path(
    r"C:\Users\Admin\Videos\videos\b9cc7a86cb57a79164acc05df3a75f16_1790343864860.mp4"
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
        reason="Orchestrator local routing test.",
        title="Orchestrator Local Routing Test",
    )

    clip_manifest = ClipManifest(
        source=source,
        selected_clips=[decision],
    )

    output_path = (
        workspace.clips_dir
        / "clip_01_Orchestrator Local Routing Test.mp4"
    )

    acquired_path = (
        workspace.source_dir
        / "section_01.mp4"
    )

    # Force the test to exercise acquisition + rendering.
    if output_path.exists():
        output_path.unlink()

    if acquired_path.exists():
        acquired_path.unlink()

    orchestrator = PipelineOrchestrator(
        intelligence_engine=object(),
    )

    print("=" * 60)
    print("ORCHESTRATOR LOCAL ROUTING TEST")
    print("=" * 60)
    print(f"Source:     {SOURCE_PATH}")
    print(f"Workspace:  {workspace.root_dir}")
    print("Range:      100.00s → 110.00s")
    print()

    results = orchestrator._ensure_rendered_clips(
        source=source,
        clip_manifest=clip_manifest,
        workspace=workspace,
    )

    if len(results) != 1:
        raise AssertionError(
            f"Expected exactly 1 rendered clip, got {len(results)}."
        )

    rendered = results[0]

    if not Path(rendered.file_path).exists():
        raise AssertionError(
            f"Rendered clip does not exist: {rendered.file_path}"
        )

    print()
    print("=" * 60)
    print("ORCHESTRATOR ROUTING SUCCEEDED")
    print("=" * 60)
    print(f"File: {rendered.file_path}")
    print(f"Size: {Path(rendered.file_path).stat().st_size:,} bytes")
    print()
    print(
        "Successfully routed:"
        "\n  Orchestrator"
        "\n      ↓"
        "\n  LocalSourceProvider"
        "\n      ↓"
        "\n  FFmpegRenderer"
        "\n      ↓"
        "\n  Final clip"
    )


if __name__ == "__main__":
    main()