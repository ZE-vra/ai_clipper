import shutil
from pathlib import Path

from src.config import Config
from src.intelligence.base import BaseAIDirector
from src.pipeline.orchestrator import PipelineOrchestrator
from src.planning.clip_planner import ClipPlanner
from src.schemas import CandidateEvaluation, EvaluationManifest


TEST_VIDEO = Path(
    r"C:\Users\Admin\Videos\videos\b9cc7a86cb57a79164acc05df3a75f16_1790343864860.mp4"
)


class FakeAIDirector(BaseAIDirector):
    """
    Deterministic AI replacement for the integration test.

    Candidate 1 receives the highest score so the real planner has
    a clear candidate to select without making a Gemini API call.
    """

    def __init__(self):
        self.call_count = 0

    def evaluate_candidates(self, candidate_manifest):
        self.call_count += 1

        evaluations = []

        for candidate in candidate_manifest.candidates:
            if candidate.candidate_id == 1:
                score = 10.0
                reason = "Deterministic integration-test winner."
                title = "Full Local Pipeline Test"
            else:
                score = 1.0
                reason = "Deterministically deprioritized for testing."
                title = None

            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    score=score,
                    reason=reason,
                    suggested_title=title,
                )
            )

        return EvaluationManifest(
            source=candidate_manifest.source,
            evaluations=evaluations,
        )


def main():
    print("=" * 60)
    print("FULL LOCAL PIPELINE INTEGRATION TEST")
    print("=" * 60)

    if not TEST_VIDEO.exists():
        raise FileNotFoundError(
            f"Test video does not exist:\n{TEST_VIDEO}"
        )

    print()
    print(f"Input video: {TEST_VIDEO}")

    # Use the normal project identity system.
    project_id, _ = Config.source_identity(str(TEST_VIDEO))
    workspace = Config.workspace_from_project_id(project_id)

    # Start from a clean workspace so every real stage is exercised.
    if workspace.root_dir.exists():
        print()
        print("Removing previous integration-test workspace...")
        shutil.rmtree(workspace.root_dir)

    workspace, existed = Config.get_or_create_workspace(
        str(TEST_VIDEO)
    )

    assert existed is False

    fake_ai = FakeAIDirector()

    # IMPORTANT:
    # - real ClipPlanner
    # - real renderer
    # - real local source provider
    # - real ingestion
    # - real Whisper
    # - real candidate generator
    orchestrator = PipelineOrchestrator(
        intelligence_engine=fake_ai,
        planner=ClipPlanner(),
    )

    print()
    print("=" * 60)
    print("RUNNING REAL PIPELINE")
    print("=" * 60)

    rendered_clips = orchestrator.run(
        str(TEST_VIDEO)
    )

    print()
    print("=" * 60)
    print("VERIFYING RESULTS")
    print("=" * 60)

    # ------------------------------------------------------------
    # Verify AI was called exactly once.
    # ------------------------------------------------------------

    assert fake_ai.call_count == 1

    print()
    print("AI director:")
    print("- Fake AI called exactly once")

    # ------------------------------------------------------------
    # Verify core checkpoints.
    # ------------------------------------------------------------

    audio_path = (
        workspace.audio_dir / "audio.mp3"
    )

    transcript_path = (
        workspace.transcript_dir / "transcript.json"
    )

    candidates_path = (
        workspace.candidates_dir / "candidates.json"
    )

    evaluations_path = (
        workspace.evaluations_dir / "evaluations.json"
    )

    clip_plan_path = (
        workspace.evaluations_dir / "clip_plan.json"
    )

    assert audio_path.exists()
    assert transcript_path.exists()
    assert candidates_path.exists()
    assert evaluations_path.exists()
    assert clip_plan_path.exists()

    print()
    print("Pipeline checkpoints:")
    print(f"- Audio:       {audio_path}")
    print(f"- Transcript:  {transcript_path}")
    print(f"- Candidates:  {candidates_path}")
    print(f"- Evaluations: {evaluations_path}")
    print(f"- Clip plan:   {clip_plan_path}")

    # ------------------------------------------------------------
    # Verify at least one real candidate was generated.
    # ------------------------------------------------------------

    candidate_files = list(
        workspace.candidates_dir.glob("*.json")
    )

    assert candidate_files

    print()
    print("Candidate generation:")
    print("- Candidate manifest exists")

    # ------------------------------------------------------------
    # Verify source sections were actually acquired.
    # ------------------------------------------------------------

    source_sections = list(
        workspace.source_dir.glob("section_*.mp4")
    )

    assert source_sections

    for section in source_sections:
        assert section.exists()
        assert section.stat().st_size > 0

    print()
    print("Source acquisition:")
    print(f"- Acquired sections: {len(source_sections)}")

    for section in source_sections:
        print(
            f"  - {section.name}: "
            f"{section.stat().st_size:,} bytes"
        )

    # ------------------------------------------------------------
    # Verify final rendered clips.
    # ------------------------------------------------------------

    assert rendered_clips

    for clip in rendered_clips:
        output_path = Path(clip.file_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    print()
    print("Final rendering:")
    print(f"- Rendered clips: {len(rendered_clips)}")

    for clip in rendered_clips:
        output_path = Path(clip.file_path)

        print(
            f"  - {output_path.name}: "
            f"{output_path.stat().st_size:,} bytes"
        )

    # ------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("FULL LOCAL PIPELINE TEST PASSED")
    print("=" * 60)

    print()
    print("Verified:")
    print("- Real local source ingestion")
    print("- Real FFmpeg audio extraction")
    print("- Real Whisper transcription")
    print("- Real candidate generation")
    print("- AI director interface")
    print("- Real clip planning")
    print("- Real local source-section acquisition")
    print("- Real FFmpeg rendering")
    print("- Real final media output")
    print("- Final rendered clips exist and contain data")


if __name__ == "__main__":
    main()