from __future__ import annotations

from pathlib import Path

from src.thumbnails.perception.frame_candidate_scorer import FrameCandidateScorer
from src.thumbnails.perception.frame_perception_analyzer import FramePerceptionAnalyzer
from src.thumbnails.perception.frame_selector import FrameSelector
from src.thumbnails.perception.local_subject_analyzer import LocalSubjectAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAME_DIR = PROJECT_ROOT / "debug" / "quality"

FRAME_NAMES = (
    "frame_05.jpg",
    "frame_15.jpg",
    "frame_30.jpg",
    "frame_45.jpg",
)


def main() -> None:
    subject_analyzer = LocalSubjectAnalyzer()
    perception_analyzer = FramePerceptionAnalyzer(
        subject_analyzer=subject_analyzer,
    )
    scorer = FrameCandidateScorer()
    selector = FrameSelector(scorer=scorer)

    candidates = []

    print("\nFRAME SELECTION EVALUATION")
    print("=" * 70)

    for frame_name in FRAME_NAMES:
        frame_path = FRAME_DIR / frame_name

        if not frame_path.is_file():
            raise FileNotFoundError(
                f"Expected frame does not exist: {frame_path}"
            )

        perception = perception_analyzer.analyze(frame_path)
        score = scorer.score(perception)

        candidates.append(perception)

        print(f"\n{frame_name}")
        print("-" * 70)
        print(f"Quality:       {score.quality_score:.3f}")
        print(f"Subject:       {score.subject_score:.3f}")
        print(f"Focal:         {score.focal_score:.3f}")
        print(f"Overall:       {score.overall_score:.3f}")
        print(f"Subjects:      {len(perception.subjects.subjects)}")
        print(f"Focal regions: {len(perception.focal.regions)}")

    selected = selector.select(candidates)

    print("\n")
    print("=" * 70)
    print("SELECTED FRAME")
    print("=" * 70)
    print(f"Frame:         {selected.perception.frame_path.name}")
    print(f"Overall score: {selected.score.overall_score:.3f}")
    print(f"Quality:       {selected.score.quality_score:.3f}")
    print(f"Subject:       {selected.score.subject_score:.3f}")
    print(f"Focal:         {selected.score.focal_score:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()