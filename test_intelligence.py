"""Test Gemini intelligence and persist its evaluation artifact."""

from pathlib import Path

from src.intelligence.gemini import GeminiDirector
from src.persistence.manifests import (
    load_candidate_manifest,
    save_evaluation_manifest,
)


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


def main():
    candidate_manifest = (
        load_candidate_manifest(
            CANDIDATES_FILE
        )
    )

    print(
        "=== GEMINI INTELLIGENCE TEST ==="
    )
    print(
        f"Video:      "
        f"{candidate_manifest.source.title}"
    )
    print(
        f"Candidates: "
        f"{candidate_manifest.total_candidates}"
    )
    print()

    director = GeminiDirector(
        model="gemini-3.8-flash",
        batch_size=40,
    )

    evaluation_manifest = (
        director.evaluate_candidates(
            candidate_manifest
        )
    )

    save_evaluation_manifest(
        evaluation_manifest,
        EVALUATIONS_FILE,
    )

    print()
    print(
        "=== EVALUATIONS SAVED ==="
    )
    print()

    print(
        f"Evaluations: "
        f"{len(evaluation_manifest.evaluations)}"
    )

    print(
        f"File: "
        f"{EVALUATIONS_FILE}"
    )

    print()

    evaluations = sorted(
        evaluation_manifest.evaluations,
        key=lambda evaluation: evaluation.score,
        reverse=True,
    )

    for evaluation in evaluations:
        print(
            f"Candidate "
            f"{evaluation.candidate_id:02d} | "
            f"Score: "
            f"{evaluation.score:.1f}/10"
        )

        print(
            f"Title: "
            f"{evaluation.suggested_title or 'None'}"
        )

        print(
            f"Reason: "
            f"{evaluation.reason}"
        )

        print("-" * 70)


if __name__ == "__main__":
    main()