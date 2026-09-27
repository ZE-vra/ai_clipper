"""Test that persisted source metadata is reused during resume."""

from pathlib import Path
import shutil

import src.pipeline.orchestrator as orchestrator_module

from src.config import Config
from src.intelligence.base import BaseAIDirector
from src.pipeline.orchestrator import PipelineOrchestrator
from src.schemas import (
    CandidateEvaluation,
    CandidateManifest,
    CandidateWindow,
    ClipDecision,
    ClipManifest,
    EvaluationManifest,
    Transcript,
    TranscriptSegment,
    VideoSource,
)


TEST_URL = (
    "https://www.youtube.com/watch?v=sourceMeta123"
)


class FakeAIDirector(BaseAIDirector):
    """Fake AI director for the metadata resume test."""

    def evaluate_candidates(
        self,
        candidate_manifest: CandidateManifest,
    ) -> EvaluationManifest:
        evaluations = []

        for candidate in (
            candidate_manifest.candidates
        ):
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    score=8.5,
                    reason="Metadata resume test.",
                    suggested_title="Metadata Test Clip",
                )
            )

        return EvaluationManifest(
            source=candidate_manifest.source,
            evaluations=evaluations,
        )


class FakePlanner:
    """Minimal planner for the test."""

    def plan(
        self,
        candidate_manifest: CandidateManifest,
        evaluation_manifest: EvaluationManifest,
    ) -> ClipManifest:
        return ClipManifest(
            source=candidate_manifest.source,
            selected_clips=[
                ClipDecision(
                    clip_id=1,
                    candidate_id=1,
                    snapped_start_time=0.0,
                    snapped_end_time=30.0,
                    final_score=8.5,
                    reason="Metadata test clip.",
                    title="Metadata Test Clip",
                )
            ],
        )


class FakeRenderer:
    """Fake renderer that creates a local output file."""

    def render_clip(
        self,
        source_url: str,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            b"fake rendered mp4"
        )

        return output_path


def fake_ingest_audio(
    source_location: str,
    workspace,
):
    audio_path = (
        workspace.audio_dir
        / "audio.mp3"
    )

    audio_path.write_bytes(
        b"fake audio"
    )

    source = VideoSource(
        source_type="youtube",
        location=source_location,
        title="PERSISTED TITLE TEST",
    )

    return source, audio_path


def fake_transcribe_audio(
    audio_path: Path,
    source: VideoSource,
    model_name: str,
):
    segment = TranscriptSegment(
        id=1,
        start=0.0,
        end=30.0,
        text="Metadata persistence test transcript.",
    )

    return Transcript(
        source=source,
        duration=30.0,
        language="en",
        segments=[segment],
    )


def fake_generate_candidates(
    transcript: Transcript,
):
    candidate = CandidateWindow(
        candidate_id=1,
        start_time=0.0,
        end_time=30.0,
        transcript_text=transcript.segments[0].text,
        segments=transcript.segments,
    )

    return CandidateManifest(
        source=transcript.source,
        total_candidates=1,
        candidates=[candidate],
    )


def main() -> None:
    print("=== SOURCE METADATA RESUME TEST ===")
    print()

    workspace, _ = (
        Config.get_or_create_workspace(
            TEST_URL
        )
    )

    if workspace.root_dir.exists():
        shutil.rmtree(
            workspace.root_dir
        )

    Config.get_or_create_workspace(
        TEST_URL
    )

    original_ingest = (
        orchestrator_module.ingest_audio
    )

    original_transcribe = (
        orchestrator_module.transcribe_audio
    )

    original_candidates = (
        orchestrator_module.generate_candidate_windows
    )

    orchestrator_module.ingest_audio = (
        fake_ingest_audio
    )

    orchestrator_module.transcribe_audio = (
        fake_transcribe_audio
    )

    orchestrator_module.generate_candidate_windows = (
        fake_generate_candidates
    )

    try:
        ai_director = FakeAIDirector()
        planner = FakePlanner()
        renderer = FakeRenderer()

        print("RUN 1")
        print("-" * 40)

        first = PipelineOrchestrator(
            ai_director=ai_director,
            planner=planner,
            renderer=renderer,
        )

        first_workspace, _, _ = first.run(
            TEST_URL
        )

        source_file = (
            first_workspace.source_dir
            / "source.json"
        )

        assert source_file.exists()

        print()
        print(
            "✓ source.json was created."
        )

        print()
        print("RUN 2")
        print("-" * 40)

        second = PipelineOrchestrator(
            ai_director=ai_director,
            planner=planner,
            renderer=renderer,
        )

        second_workspace, _, _ = second.run(
            TEST_URL
        )

        assert second_workspace.root_dir == (
            first_workspace.root_dir
        )

        assert second.source is not None

        assert second.source.title == (
            "PERSISTED TITLE TEST"
        )

        assert second.source.location == (
            TEST_URL
        )

        assert second.source.source_type == (
            "youtube"
        )

        print()
        print(
            "✓ Persisted source title was loaded."
        )

        print(
            "✓ Persisted source URL was loaded."
        )

        print(
            "✓ Persisted source type was loaded."
        )

        print()
        print("=" * 60)
        print("SOURCE METADATA TEST PASSED")
        print("=" * 60)

    finally:
        orchestrator_module.ingest_audio = (
            original_ingest
        )

        orchestrator_module.transcribe_audio = (
            original_transcribe
        )

        orchestrator_module.generate_candidate_windows = (
            original_candidates
        )


if __name__ == "__main__":
    main()