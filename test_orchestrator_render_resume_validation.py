from pathlib import Path

from src.config import Config
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
        reason="Orchestrator resume validation test.",
        title="Orchestrator Resume Validation Test",
    )

    clip_manifest = ClipManifest(
        source=source,
        selected_clips=[decision],
    )

    output_path = (
        workspace.clips_dir
        / "clip_01_Orchestrator Resume Validation Test.mp4"
    )

    acquired_path = (
        workspace.source_dir
        / "section_01.mp4"
    )

    if output_path.exists():
        output_path.unlink()

    if acquired_path.exists():
        acquired_path.unlink()

    orchestrator = PipelineOrchestrator(
        intelligence_engine=object(),
    )

    print("=" * 60)
    print("ORCHESTRATOR RESUME VALIDATION TEST")
    print("=" * 60)
    print(f"Source:     {SOURCE_PATH}")
    print(f"Workspace:  {workspace.root_dir}")
    print("Range:      100.00s → 110.00s")
    print()

    # ---------------------------------------------------------
    # STEP 1
    # Create a valid rendered clip.
    # ---------------------------------------------------------

    print("=" * 60)
    print("STEP 1: CREATE VALID CLIP")
    print("=" * 60)

    first_results = orchestrator._ensure_rendered_clips(
        source=source,
        clip_manifest=clip_manifest,
        workspace=workspace,
    )

    if len(first_results) != 1:
        raise AssertionError(
            f"Expected exactly 1 rendered clip, got {len(first_results)}."
        )

    if not output_path.exists():
        raise AssertionError(
            "Expected the first render to create the output file."
        )

    print()
    print(f"Initial clip created: {output_path}")
    print(f"Initial size: {output_path.stat().st_size:,} bytes")
    print()

    # ---------------------------------------------------------
    # STEP 2
    # Corrupt the existing output.
    # ---------------------------------------------------------

    print("=" * 60)
    print("STEP 2: CORRUPT EXISTING CLIP")
    print("=" * 60)

    output_path.write_bytes(
        b"This is deliberately corrupted media."
    )

    corrupted_size = output_path.stat().st_size

    print()
    print(
        f"Existing clip deliberately corrupted."
    )
    print(f"Corrupted size: {corrupted_size:,} bytes")
    print()

    # ---------------------------------------------------------
    # STEP 3
    # Run the orchestrator again.
    #
    # It must detect the invalid existing output,
    # delete it, reacquire the source section,
    # and render a fresh valid clip.
    # ---------------------------------------------------------

    print("=" * 60)
    print("STEP 3: RESUME PIPELINE")
    print("=" * 60)

    second_results = orchestrator._ensure_rendered_clips(
        source=source,
        clip_manifest=clip_manifest,
        workspace=workspace,
    )

    if len(second_results) != 1:
        raise AssertionError(
            f"Expected exactly 1 rendered clip after resume, "
            f"got {len(second_results)}."
        )

    if not output_path.exists():
        raise AssertionError(
            "Expected the invalid output to be replaced by a new render."
        )

    # ---------------------------------------------------------
    # STEP 4
    # Explicitly validate the replacement.
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 4: VALIDATE REPLACEMENT")
    print("=" * 60)

    orchestrator.renderer.validate_media(output_path)

    final_size = output_path.stat().st_size

    if final_size <= corrupted_size:
        raise AssertionError(
            "Replacement output was not larger than the deliberately "
            "corrupted file."
        )

    print()
    print("=" * 60)
    print("RESUME VALIDATION SUCCEEDED")
    print("=" * 60)
    print(f"File: {output_path}")
    print(f"Final size: {final_size:,} bytes")
    print()
    print(
        "Successfully verified:"
        "\n  Existing output"
        "\n      ↓"
        "\n  Validation failed"
        "\n      ↓"
        "\n  Invalid output removed"
        "\n      ↓"
        "\n  Source reacquired"
        "\n      ↓"
        "\n  Clip re-rendered"
        "\n      ↓"
        "\n  Final output validated"
    )


if __name__ == "__main__":
    main()