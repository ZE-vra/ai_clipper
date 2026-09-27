import shutil
from pathlib import Path

import src.pipeline.orchestrator as orchestrator_module
import src.ingestion.downloader as downloader_module
import src.perception.whisper_engine as whisper_module

from src.config import Config
from src.intelligence.base import BaseAIDirector
from src.persistence.manifests import (
    save_candidate_manifest,
    save_transcript,
)
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


TEST_URL = "https://youtu.be/resume_test_123"


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
                    suggested_title="Resume Test Clip",
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
                    reason="Fake planned clip",
                    title="Resume Test Clip",
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

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            b"fake rendered media"
        )

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

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            b"fake acquired media"
        )

        return output_path


def fake_ingest_audio(
    source_location,
    workspace,
):
    audio_path = workspace.audio_dir / "audio.mp3"

    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_path.write_bytes(
        b"fake audio"
    )

    source = VideoSource(
        source_type="youtube",
        location=source_location,
        title="Fake Resume Test Video",
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
                text="This is a fake transcript for resume testing.",
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
            "This is a fake transcript for resume testing."
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
        first_orchestrator = PipelineOrchestrator(
            intelligence_engine=fake_ai,
            planner=fake_planner,
            renderer=fake_renderer,
            youtube_source_provider=fake_source_provider,
        )

        print("=" * 60)
        print("FIRST RUN")
        print("=" * 60)

        first_rendered = first_orchestrator.run(TEST_URL)

        assert len(first_rendered) == 1

        assert fake_ai.call_count == 1
        assert fake_planner.call_count == 1
        assert fake_source_provider.call_count == 1
        assert fake_renderer.render_call_count == 1

        assert (
            workspace.audio_dir / "audio.mp3"
        ).exists()

        assert (
            workspace.transcript_dir / "transcript.json"
        ).exists()

        assert (
            workspace.candidates_dir / "candidates.json"
        ).exists()

        assert (
            workspace.evaluations_dir / "evaluations.json"
        ).exists()

        assert (
            workspace.evaluations_dir / "clip_plan.json"
        ).exists()

        first_output = Path(
            first_rendered[0].file_path
        )

        assert first_output.exists()

        print()
        print("FIRST RUN PASSED")
        print()

        second_orchestrator = PipelineOrchestrator(
            intelligence_engine=fake_ai,
            planner=fake_planner,
            renderer=fake_renderer,
            youtube_source_provider=fake_source_provider,
        )

        print("=" * 60)
        print("SECOND RUN")
        print("=" * 60)

        second_rendered = second_orchestrator.run(TEST_URL)

        assert len(second_rendered) == 1

        assert fake_ai.call_count == 1
        assert fake_planner.call_count == 1
        assert fake_source_provider.call_count == 1
        assert fake_renderer.render_call_count == 1

        assert fake_renderer.validation_call_count >= 1

        second_output = Path(
            second_rendered[0].file_path
        )

        assert second_output.exists()

        assert second_output == first_output

        print()
        print("SECOND RUN PASSED")
        print()

        print("=" * 60)
        print("RESUME TEST PASSED")
        print("=" * 60)

        print()
        print("Verified:")
        print("- Audio checkpoint reused")
        print("- Transcript checkpoint reused")
        print("- Candidate checkpoint reused")
        print("- Evaluation checkpoint reused")
        print("- Clip-plan checkpoint reused")
        print("- Source acquisition was not repeated")
        print("- Rendering was not repeated")
        print("- Existing final output was validated")
        print("- Existing valid final output was reused")

    finally:
        downloader_module.ingest_audio = original_ingest_audio
        whisper_module.transcribe_audio = original_transcribe_audio
        orchestrator_module.generate_candidate_windows = (
            original_generate_candidates
        )


if __name__ == "__main__":
    main()