import json
from pathlib import Path

from src.intelligence.gemini import GeminiDirector
from src.planning.clip_planner import ClipPlanner
from src.schemas import (
    CandidateManifest,
    CandidateWindow,
    TranscriptSegment,
    VideoSource,
)


PROJECT_DIR = Path(
    r".\projects\2026-09-26_17-29-08_outube_com_watch_v_dQw4w9WgXcQ"
)

CANDIDATES_FILE = (
    PROJECT_DIR
    / "candidates"
    / "candidates.json"
)


def load_candidate_manifest() -> CandidateManifest:
    """Load the previously generated candidate manifest."""

    with open(
        CANDIDATES_FILE,
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    source_data = data["source"]

    source = VideoSource(
        source_type=source_data["source_type"],
        location=source_data["location"],
        title=source_data.get("title"),
    )

    candidates = []

    for item in data["candidates"]:
        segments = [
            TranscriptSegment(
                id=segment["id"],
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
            )
            for segment in item.get("segments", [])
        ]

        candidates.append(
            CandidateWindow(
                candidate_id=item["candidate_id"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                transcript_text=item["transcript_text"],
                segments=segments,
            )
        )

    return CandidateManifest(
        source=source,
        total_candidates=len(candidates),
        candidates=candidates,
    )


def main():
    # ---------------------------------------------------------
    # 1. Load candidates
    # ---------------------------------------------------------

    candidate_manifest = load_candidate_manifest()

    print("=== AI → PLANNER TEST ===")
    print()
    print(f"Video:      {candidate_manifest.source.title}")
    print(f"Candidates: {candidate_manifest.total_candidates}")
    print()

    # ---------------------------------------------------------
    # 2. Gemini evaluates candidates
    # ---------------------------------------------------------

    director = GeminiDirector(
        model="gemini-3.8-flash",
        batch_size=40,
    )

    evaluation_manifest = director.evaluate_candidates(
        candidate_manifest
    )

    print()
    print(
        f"Gemini evaluations: "
        f"{len(evaluation_manifest.evaluations)}"
    )

    # ---------------------------------------------------------
    # 3. Planner selects final candidates
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 4. Display final plan
    # ---------------------------------------------------------

    print()
    print("=== FINAL CLIP PLAN ===")
    print()

    if not clip_manifest.selected_clips:
        print("No clips were selected.")
        return

    for clip in clip_manifest.selected_clips:
        print(
            f"Clip {clip.clip_id:02d}"
        )

        print(
            f"  Candidate: "
            f"{clip.candidate_id}"
        )

        print(
            f"  Time: "
            f"{clip.snapped_start_time:.2f}s "
            f"→ "
            f"{clip.snapped_end_time:.2f}s"
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
            f"{clip.title or 'Untitled'}"
        )

        print(
            f"  Reason: "
            f"{clip.reason}"
        )

        print("-" * 70)

    # ---------------------------------------------------------
    # 5. Save the plan
    # ---------------------------------------------------------

    plan_file = (
        PROJECT_DIR
        / "clip_plan.json"
    )

    with open(
        plan_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            clip_manifest.to_json()
        )

    print()
    print(
        f"Saved plan: {plan_file}"
    )


if __name__ == "__main__":
    main()