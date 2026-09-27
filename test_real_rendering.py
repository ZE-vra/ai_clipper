"""Render the real clips selected by the saved clip plan."""

from pathlib import Path

from src.config import Config
from src.persistence.manifests import load_clip_manifest
from src.pipeline.artifact_state import ArtifactStateChecker
from src.rendering.ffmpeg_renderer import FFmpegRenderer


VIDEO_URL = (
    "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"
)


def main() -> None:
    print("=== REAL VIDEO RENDERING TEST ===")
    print()

    workspace, existed = Config.get_or_create_workspace(
        VIDEO_URL
    )

    print(f"Project: {workspace.project_id}")
    print(f"Existing project: {existed}")
    print()

    print("[1/2] LOAD CLIP PLAN")
    print("-" * 40)

    clip_plan_path = (
        workspace.evaluations_dir
        / "clip_plan.json"
    )

    if not clip_plan_path.exists():
        raise FileNotFoundError(
            "clip_plan.json does not exist. "
            "Run test_real_planner.py first."
        )

    clip_manifest = load_clip_manifest(
        workspace
    )

    print(
        f"✓ Loaded "
        f"{len(clip_manifest.selected_clips)} selected clips."
    )
    print()

    print("[2/2] RENDER SELECTED CLIPS")
    print("-" * 40)

    renderer = FFmpegRenderer()

    artifact_checker = ArtifactStateChecker(
        workspace
    )

    rendered = []

    for clip in clip_manifest.selected_clips:
        title = (
            clip.title
            or f"clip_{clip.clip_id:02d}"
        )

        output_path = artifact_checker.rendered_clip_path(
            clip_id=clip.clip_id,
            title=title,
        )

        print()
        print(
            f"Rendering Clip {clip.clip_id:02d}"
        )

        print(
            f"Candidate: "
            f"{clip.candidate_id:02d}"
        )

        print(
            f"Time: "
            f"{clip.snapped_start_time:.2f}s"
            f" -> "
            f"{clip.snapped_end_time:.2f}s"
        )

        print(
            f"Duration: "
            f"{clip.duration:.2f}s"
        )

        print(
            f"Output: "
            f"{output_path}"
        )

        if output_path.exists():
            print(
                "✓ Existing rendered clip found. "
                "Reusing it."
            )
        else:
            renderer.render_clip(
                source_url=VIDEO_URL,
                output_path=output_path,
                start_time=clip.snapped_start_time,
                end_time=clip.snapped_end_time,
            )

            print("✓ Clip rendered successfully.")

        rendered.append(output_path)

    print()
    print("=" * 70)
    print("RENDERING RESULTS")
    print("=" * 70)

    for path in rendered:
        print()
        print(f"✓ {path}")
        print(f"  Exists: {path.exists()}")

        if path.exists():
            size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"  Size: "
                f"{size_mb:.2f} MB"
            )

    print()
    print("=" * 70)
    print("REAL VIDEO RENDERING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()