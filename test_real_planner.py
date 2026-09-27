"""Run the real Clip Planner against saved Gemini evaluations."""

from src.config import Config
from src.persistence.manifests import (
    load_candidate_manifest,
    load_evaluation_manifest,
    save_clip_manifest,
)
from src.planning.clip_planner import ClipPlanner


VIDEO_URL = (
    "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"
)


def main() -> None:
    print("=== REAL CLIP PLANNER TEST ===")
    print()

    workspace, existed = Config.get_or_create_workspace(
        VIDEO_URL
    )

    print(f"Project: {workspace.project_id}")
    print(f"Existing project: {existed}")
    print()

    print("[1/3] LOAD CANDIDATES")
    print("-" * 40)

    candidate_manifest = load_candidate_manifest(
        workspace
    )

    print(
        f"✓ Loaded "
        f"{len(candidate_manifest.candidates)} candidates."
    )
    print()

    print("[2/3] LOAD GEMINI EVALUATIONS")
    print("-" * 40)

    evaluation_manifest = load_evaluation_manifest(
        workspace
    )

    print(
        f"✓ Loaded "
        f"{len(evaluation_manifest.evaluations)} evaluations."
    )
    print()

    print("[3/3] RUN CLIP PLANNER")
    print("-" * 40)

    planner = ClipPlanner(
        min_score=7.0,
        max_clips=10,
        min_duration=15.0,
        max_duration=180.0,
        overlap_threshold=0.5,
    )

    clip_manifest = planner.plan(
        candidate_manifest=candidate_manifest,
        evaluation_manifest=evaluation_manifest,
    )

    save_clip_manifest(
        clip_manifest,
        workspace,
    )

    print(
        f"✓ Planner selected "
        f"{len(clip_manifest.selected_clips)} clips."
    )

    print()
    print("=" * 70)
    print("REAL CLIP PLAN")
    print("=" * 70)

    if not clip_manifest.selected_clips:
        print()
        print("No clips were selected.")
    else:
        for clip in clip_manifest.selected_clips:
            print()
            print(
                f"Clip {clip.clip_id:02d}"
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
                f"Score: "
                f"{clip.final_score:.1f}/10"
            )

            print(
                f"Title: "
                f"{clip.title}"
            )

            print(
                f"Reason: "
                f"{clip.reason}"
            )

    print()
    print("=" * 70)
    print("REAL CLIP PLANNER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()