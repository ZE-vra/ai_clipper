from pathlib import Path

from src.config import Config
from src.intelligence.base import BaseAIDirector
from src.persistence.manifests import load_source
from src.pipeline.orchestrator import PipelineOrchestrator
from src.rendering.ffmpeg_renderer import FFmpegRenderer
from src.rendering.source_provider import (
    LocalSourceProvider,
    YouTubeSourceProvider,
)


INPUT_VIDEO = Path(
    r"C:\Users\Admin\Videos\videos\b9cc7a86cb57a79164acc05df3a75f16_1790343864860.mp4"
)


class ForbiddenCallError(RuntimeError):
    """Raised if a stage that should be skipped is executed."""


class ResumeGuardAI(BaseAIDirector):
    def evaluate_candidates(self, candidate_manifest):
        raise ForbiddenCallError(
            "FAIL: Gemini/AI evaluation was executed during resume."
        )


class ResumeGuardPlanner:
    def plan(self, candidate_manifest, evaluation_manifest):
        raise ForbiddenCallError(
            "FAIL: Clip planning was executed during resume."
        )


class ResumeGuardSourceProvider(LocalSourceProvider):
    def acquire_section(
        self,
        source_path,
        output_path,
        start_time,
        end_time,
    ):
        raise ForbiddenCallError(
            "FAIL: Source acquisition was executed during resume."
        )


class ResumeGuardRenderer:
    def __init__(self):
        self.real_renderer = FFmpegRenderer()

    def validate_media(self, media_path):
        return self.real_renderer.validate_media(media_path)

    def render_clip(
        self,
        source_path,
        output_path,
        start_time,
        end_time,
    ):
        raise ForbiddenCallError(
            "FAIL: FFmpeg rendering was executed during resume."
        )


def forbidden_ingest_audio(*args, **kwargs):
    raise ForbiddenCallError(
        "FAIL: Audio ingestion was executed during resume."
    )


def forbidden_transcribe_audio(*args, **kwargs):
    raise ForbiddenCallError(
        "FAIL: Whisper transcription was executed during resume."
    )


def forbidden_generate_candidates(*args, **kwargs):
    raise ForbiddenCallError(
        "FAIL: Candidate generation was executed during resume."
    )


def main():
    print("=" * 60)
    print("FULL LOCAL PIPELINE RESUME TEST")
    print("=" * 60)

    if not INPUT_VIDEO.exists():
        raise RuntimeError(
            f"Input video does not exist:\n{INPUT_VIDEO}"
        )

    workspace = Config.find_workspace(str(INPUT_VIDEO))

    if workspace is None:
        raise RuntimeError(
            "The integration-test workspace does not exist.\n"
            "Run test_full_local_pipeline.py once first."
        )

    print()
    print(f"Input video: {INPUT_VIDEO}")
    print(f"Workspace:   {workspace.root_dir}")

    required_artifacts = [
        workspace.audio_dir / "audio.mp3",
        workspace.transcript_dir / "transcript.json",
        workspace.candidates_dir / "candidates.json",
        workspace.evaluations_dir / "evaluations.json",
        workspace.evaluations_dir / "clip_plan.json",
    ]

    print()
    print("Checking existing checkpoints...")

    for artifact in required_artifacts:
        if not artifact.exists():
            raise RuntimeError(
                f"Required checkpoint is missing:\n{artifact}\n\n"
                "Run test_full_local_pipeline.py first."
            )

        print(f"FOUND: {artifact}")

    rendered_clips = list(workspace.clips_dir.glob("*.mp4"))

    if not rendered_clips:
        raise RuntimeError(
            "No rendered clips exist in the workspace.\n"
            "Run test_full_local_pipeline.py first."
        )

    print()
    print("Existing rendered clips:")

    for clip in rendered_clips:
        print(f"FOUND: {clip}")

    print()
    print("=" * 60)
    print("INSTALLING RESUME GUARDS")
    print("=" * 60)

    # Import the actual modules so we can guard the expensive stages.
    import src.ingestion.downloader as downloader
    import src.perception.whisper_engine as whisper_engine
    import src.pipeline.orchestrator as orchestrator_module

    original_ingest_audio = downloader.ingest_audio
    original_transcribe_audio = whisper_engine.transcribe_audio
    original_generate_candidates = (
        orchestrator_module.generate_candidate_windows
    )

    downloader.ingest_audio = forbidden_ingest_audio
    whisper_engine.transcribe_audio = forbidden_transcribe_audio
    orchestrator_module.generate_candidate_windows = (
        forbidden_generate_candidates
    )

    try:
        print("Audio ingestion guard: ENABLED")
        print("Whisper guard: ENABLED")
        print("Candidate generation guard: ENABLED")
        print("AI evaluation guard: ENABLED")
        print("Clip planning guard: ENABLED")
        print("Source acquisition guard: ENABLED")
        print("Rendering guard: ENABLED")

        print()
        print("=" * 60)
        print("RUNNING RESUME")
        print("=" * 60)

        source = load_source(workspace)

        source_provider = ResumeGuardSourceProvider()

        orchestrator = PipelineOrchestrator(
            intelligence_engine=ResumeGuardAI(),
            planner=ResumeGuardPlanner(),
            renderer=ResumeGuardRenderer(),
            youtube_source_provider=YouTubeSourceProvider(),
            local_source_provider=source_provider,
        )

        results = orchestrator.run(str(INPUT_VIDEO))

        print()
        print("=" * 60)
        print("VERIFYING RESUME")
        print("=" * 60)

        if not results:
            raise RuntimeError(
                "FAIL: Pipeline returned no rendered clips."
            )

        print(f"Returned clips: {len(results)}")

        for result in results:
            output_path = Path(result.file_path)

            if not output_path.exists():
                raise RuntimeError(
                    f"FAIL: Returned clip does not exist:\n{output_path}"
                )

            if output_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"FAIL: Returned clip is empty:\n{output_path}"
                )

            print()
            print(f"Clip ID: {result.clip_id}")
            print(f"Title:   {result.title}")
            print(f"Path:    {output_path}")
            print(f"Size:    {output_path.stat().st_size:,} bytes")

        print()
        print("=" * 60)
        print("FULL LOCAL PIPELINE RESUME TEST PASSED")
        print("=" * 60)

        print()
        print("Verified:")
        print("- Existing audio checkpoint was reused")
        print("- Existing transcript checkpoint was reused")
        print("- Existing candidate checkpoint was reused")
        print("- Existing evaluation checkpoint was reused")
        print("- Existing clip plan was reused")
        print("- Existing final clip was validated")
        print("- Existing final clip was reused")
        print("- Whisper was NOT executed")
        print("- Candidate generation was NOT executed")
        print("- Gemini was NOT executed")
        print("- Clip planning was NOT executed")
        print("- Source acquisition was NOT executed")
        print("- FFmpeg rendering was NOT executed")

    finally:
        downloader.ingest_audio = original_ingest_audio
        whisper_engine.transcribe_audio = original_transcribe_audio
        orchestrator_module.generate_candidate_windows = (
            original_generate_candidates
        )


if __name__ == "__main__":
    main()