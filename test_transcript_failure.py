import shutil

import src.ingestion.downloader as downloader_module
import src.perception.whisper_engine as whisper_module
import src.pipeline.orchestrator as orchestrator_module

from src.config import Config
from src.intelligence.base import BaseAIDirector
from src.persistence.manifests import save_candidate_manifest, save_transcript
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


TEST_URL = "https://youtu.be/transcript_failure_123"


class FakeAIDirector(BaseAIDirector):
    def __init__(self):
        self.call_count = 0

    def evaluate_candidates(self, candidate_manifest):
        self.call_count += 1

        return EvaluationManifest(
            source=candidate_manifest.source,
            evaluations=[
                CandidateEvaluation(
                    candidate_id=1,
                    score=8.0,
                    reason="Fake evaluation",
                    suggested_title="Transcript Recovery Test",
                )
            ],
        )


class FakePlanner:
    def __init__(self):
        self.call_count = 0

    def plan(self, candidate_manifest, evaluation_manifest):
        self.call_count += 1

        return ClipManifest(
            source=candidate_manifest.source,
            selected_clips=[
                ClipDecision(
                    clip_id=1,
                    candidate_id=1,
                    snapped_start_time=0.0,
                    snapped_end_time=10.0,
                    final_score=8.0,
                    reason="Fake transcript recovery plan",
                    title="Transcript Recovery Test",
                )
            ],
        )


class FakeRenderer:
    def __init__(self):
        self.render_call_count = 0
        self.validation_call_count = 0

    def render_clip(
        self,
        source_path,
        output_path,
        start_time,
        end_time,
    ):
        self.render_call_count += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake rendered media")

        return output_path

    def validate_media(self, media_path):
        self.validation_call_count += 1

        if not media_path.exists():
            raise RuntimeError(
                f"Expected media to exist: {media_path}"
            )


class FakeYouTubeSourceProvider:
    def __init__(self):
        self.call_count = 0

    def acquire_section(
        self,
        source_url,
        output_path,
        start_time,
        end_time,
    ):
        self.call_count += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake acquired media")

        return output_path


def fake_ingest_audio(source_location, workspace):
    audio_path = workspace.audio_dir / "audio.mp3"

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake audio")

    source = VideoSource(
        source_type="youtube",
        location=source_location,
        title="Fake Transcript Recovery Test Video",
    )

    return source, audio_path


def fake_transcribe_audio(
    audio_path,
    source,
    workspace,
    model_name="base",
    language=None,
):
    transcript = Transcript(
        source=source,
        duration=10.0,
        language="en",
        segments=[
            TranscriptSegment(
                id=1,
                start=0.0,
                end=10.0,
                text="This transcript was regenerated successfully.",
            )
        ],
    )

    save_transcript(
        transcript,
        workspace,
    )

    return transcript


def fake_generate_candidate_windows(
    transcript,
    workspace,
    **kwargs,
):
    candidate = CandidateWindow(
        candidate_id=1,
        start_time=0.0,
        end_time=10.0,
        transcript_text=(
            "This transcript was regenerated successfully."
        ),
        segments=transcript.segments,
    )

    manifest = CandidateManifest(
        source=transcript.source,
        total_candidates=1,
        candidates=[candidate],
    )

    save_candidate_manifest(
        manifest,
        workspace,
    )

    return manifest


def main():
    workspace, _ = Config.get_or_create_workspace(TEST_URL)

    if workspace.root_dir.exists():
        shutil.rmtree(workspace.root_dir)

    workspace, existed = Config.get_or_create_workspace(TEST_URL)

    assert existed is False

    fake_ai = FakeAIDirector()
    fake_planner = FakePlanner()
    fake_renderer = FakeRenderer()
    fake_source_provider = FakeYouTubeSourceProvider()

    original_ingest_audio = downloader_module.ingest_audio
    original_transcribe_audio = whisper_module.transcribe_audio
    original_generate_candidates = (
        orchestrator_module.generate_candidate_windows
    )

    downloader_module.ingest_audio = fake_ingest_audio
    whisper_module.transcribe_audio = fake_transcribe_audio
    orchestrator_module.generate_candidate_windows = (
        fake_generate_candidate_windows
    )

    try:
        orchestrator = PipelineOrchestrator(
            intelligence_engine=fake_ai,
            planner=fake_planner,
            renderer=fake_renderer,
            youtube_source_provider=fake_source_provider,
        )

        print("=" * 60)
        print("INITIAL RUN")
        print("=" * 60)

        first_rendered = orchestrator.run(TEST_URL)

        assert len(first_rendered) == 1

        assert fake_ai.call_count == 1
        assert fake_planner.call_count == 1
        assert fake_source_provider.call_count == 1
        assert fake_renderer.render_call_count == 1

        print()
        print("INITIAL RUN PASSED")
        print()

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

        rendered_clip = next(
            workspace.clips_dir.glob("*.mp4")
        )

        source_section = next(
            workspace.source_dir.glob("section_*.mp4")
        )

        assert transcript_path.exists()
        assert candidates_path.exists()
        assert evaluations_path.exists()
        assert clip_plan_path.exists()
        assert rendered_clip.exists()
        assert source_section.exists()

        print("=" * 60)
        print("SIMULATING TRANSCRIPT FAILURE")
        print("=" * 60)
        print()

        print("Deleting transcript checkpoint:")
        print(transcript_path)

        transcript_path.unlink()

        assert not transcript_path.exists()

        print()
        print("Transcript checkpoint deleted.")
        print("All downstream artifacts deliberately left intact.")

        print()
        print("=" * 60)
        print("RECOVERY RUN")
        print("=" * 60)

        second_rendered = orchestrator.run(TEST_URL)

        assert len(second_rendered) == 1

        print()
        print("=" * 60)
        print("VERIFYING RECOVERY")
        print("=" * 60)

        # Every downstream stage must have been regenerated.
        assert fake_ai.call_count == 2
        assert fake_planner.call_count == 2
        assert fake_source_provider.call_count == 2
        assert fake_renderer.render_call_count == 2

        # The transcript itself must have been regenerated.
        assert transcript_path.exists()

        # Every downstream checkpoint must exist again.
        assert candidates_path.exists()
        assert evaluations_path.exists()
        assert clip_plan_path.exists()

        # Final media must exist again.
        assert rendered_clip.exists()
        assert source_section.exists()

        print()
        print("Recovery behavior:")
        print("- Audio was reused")
        print("- Transcript was regenerated")
        print("- Candidates were regenerated")
        print("- Evaluations were regenerated")
        print("- Clip plan was regenerated")
        print("- Source section was reacquired")
        print("- Final clip was rerendered")

        print()
        print("=" * 60)
        print("TRANSCRIPT FAILURE TEST PASSED")
        print("=" * 60)

    finally:
        downloader_module.ingest_audio = original_ingest_audio
        whisper_module.transcribe_audio = original_transcribe_audio
        orchestrator_module.generate_candidate_windows = (
            original_generate_candidates
        )


if __name__ == "__main__":
    main()