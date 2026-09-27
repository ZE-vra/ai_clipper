"""Test ClipPlanner using persisted pipeline artifacts.

This test does NOT call Gemini.

It verifies:

    candidates.json
        ↓
    evaluations.json
        ↓
    ClipPlanner
        ↓
    ClipManifest
        ↓
    clip_plan.json
"""

from pathlib import Path

from src.persistence.manifests import (
    load_candidate_manifest,
    load_evaluation_manifest,
    save_clip_manifest,
)
from src.planning.clip_planner import ClipPlanner


PROJECT_DIR = Path(
    r".\projects\2026-09-26_17-29-08_outube_com_watch_v_dQw4w9WgXcQ"
)

CANDIDATES_FILE = (
    PROJECT_DIR
    / "candidates"
    / "candidates.json"
)

EVALUATIONS_FILE = (
    PROJECT_DIR
    / "evaluations"
    / "evaluations.json"
)

PLAN_FILE = (
    PROJECT_DIR
    / "evaluations"
    / "clip_plan.json"
)


def main():
    print(
        "=== PLANNER FROM SAVED EVALUATIONS ==="
    )
    print()

    candidate_manifest = (
        load_candidate_manifest(
            CANDIDATES_FILE
        )
    )

    evaluation_manifest = (
        load_evaluation_manifest(
            EVALUATIONS_FILE
        )
    )

    print(
        f"Candidates loaded: "
        f"{len(candidate_manifest.candidates)}"
    )

    print(
        f"Evaluations loaded: "
        f"{len(evaluation_manifest.evaluations)}"
    )

    print()

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
        PLAN_FILE,
    )

    print(
        "=== FINAL CLIP PLAN ==="
    )
    print()

    print(
        f"Selected clips: "
        f"{len(clip_manifest.selected_clips)}"
    )

    print()

    for clip in clip_manifest.selected_clips:
        print(
            f"Clip {clip.clip_id:02d}"
        )

        print(
            f"  Candidate: "
            f"{clip.candidate_id}"
        )

        print(
            f"  Start: "
            f"{clip.snapped_start_time:.3f}s"
        )

        print(
            f"  End: "
            f"{clip.snapped_end_time:.3f}s"
        )

        print(
            f"  Duration: "
            f"{clip.duration:.2f}s"
        )

        print(
            f"  Score: "
            f"{clip.final_score:.1f}/10"
        )

        print(
            f"  Title: "
            f"{clip.title or 'None'}"
        )

        print(
            f"  Reason: "
            f"{clip.reason}"
        )

        print("-" * 70)

    print()

    print(
        f"Saved plan to: "
        f"{PLAN_FILE}"
    )


if __name__ == "__main__":
    main()