"""Run the real AI Director against the real candidate manifest."""

from src.config import Config
from src.intelligence.gemini import GeminiDirector
from src.persistence.manifests import (
    load_candidate_manifest,
    load_evaluation_manifest,
    save_evaluation_manifest,
)


VIDEO_URL = (
    "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"
)


def main() -> None:
    print("=== REAL AI DIRECTOR TEST ===")
    print()

    # ---------------------------------------------------------
    # WORKSPACE
    # ---------------------------------------------------------

    workspace, existed = Config.get_or_create_workspace(
        VIDEO_URL
    )

    print(f"Project: {workspace.project_id}")
    print(f"Existing project: {existed}")
    print()

    # ---------------------------------------------------------
    # LOAD CANDIDATES
    # ---------------------------------------------------------

    print("[1/2] LOAD CANDIDATES")
    print("-" * 40)

    candidate_path = (
        workspace.candidates_dir
        / "candidates.json"
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            "Candidate manifest does not exist. "
            "Run test_real_discovery.py first."
        )

    candidate_manifest = load_candidate_manifest(
        workspace
    )

    print(
        f"✓ Loaded "
        f"{len(candidate_manifest.candidates)} candidates."
    )

    print()

    # ---------------------------------------------------------
    # GEMINI EVALUATION
    # ---------------------------------------------------------

    print("[2/2] REAL GEMINI EVALUATION")
    print("-" * 40)

    evaluation_path = (
        workspace.evaluations_dir
        / "evaluations.json"
    )

    if evaluation_path.exists():
        print("✓ Existing evaluations found.")
        print("  Removing old evaluation artifact so Gemini")
        print("  evaluates the current 7 candidates.")
        print()

        evaluation_path.unlink()

    print(
        f"Sending "
        f"{len(candidate_manifest.candidates)} candidates "
        "to Gemini..."
    )
    print()

    director = GeminiDirector()

    evaluation_manifest = director.evaluate_candidates(
        candidate_manifest
    )

    save_evaluation_manifest(
        evaluation_manifest,
        workspace,
    )

    print(
        f"✓ Gemini evaluated "
        f"{len(evaluation_manifest.evaluations)} candidates."
    )

    print()

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("=" * 70)
    print("AI DIRECTOR RESULTS")
    print("=" * 70)

    evaluations_by_id = {
        evaluation.candidate_id: evaluation
        for evaluation in evaluation_manifest.evaluations
    }

    for candidate in candidate_manifest.candidates:
        evaluation = evaluations_by_id.get(
            candidate.candidate_id
        )

        print()
        print(
            f"Candidate "
            f"{candidate.candidate_id:02d}"
        )

        print(
            f"Time: "
            f"{candidate.start_time:.2f}s"
            f" -> "
            f"{candidate.end_time:.2f}s"
        )

        print(
            f"Duration: "
            f"{candidate.duration:.2f}s"
        )

        if evaluation is None:
            print("Score: MISSING")
            print("Reason: No evaluation returned.")
            print("Title: None")
            continue

        print(
            f"Score: "
            f"{evaluation.score:.1f}/10"
        )

        print(
            f"Reason: "
            f"{evaluation.reason}"
        )

        print(
            f"Suggested title: "
            f"{evaluation.suggested_title}"
        )

    print()
    print("=" * 70)
    print("REAL AI DIRECTOR TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()