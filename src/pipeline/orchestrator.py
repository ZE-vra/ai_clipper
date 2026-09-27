"""Pipeline orchestration for the resumable clipping workflow."""

from pathlib import Path
from typing import Optional

from src.candidates.window_generator import generate_candidate_windows
from src.config import Config, ProjectWorkspace
from src.exceptions import ClipperError
from src.intelligence.base import BaseAIDirector
from src.persistence.manifests import (
    load_candidate_manifest,
    load_clip_manifest,
    load_evaluation_manifest,
    load_source,
    load_transcript,
    save_clip_manifest,
    save_evaluation_manifest,
    save_source,
)
from src.pipeline.artifact_state import ArtifactStateChecker
from src.rendering.ffmpeg_renderer import (
    FFmpegRenderer,
    RenderingError,
)
from src.rendering.source_provider import (
    LocalSourceProvider,
    SourceAcquisitionError,
    YouTubeSourceProvider,
)
from src.schemas import (
    FinalRenderedClip,
    VideoSource,
)
from src.planning.clip_planner import ClipPlanner


class PipelineError(ClipperError):
    """Raised when pipeline orchestration fails."""


class PipelineOrchestrator:
    """Coordinate the complete resumable clipping pipeline."""

    def __init__(
        self,
        intelligence_engine: BaseAIDirector,
        planner: Optional[ClipPlanner] = None,
        renderer: Optional[FFmpegRenderer] = None,
        youtube_source_provider: Optional[YouTubeSourceProvider] = None,
        local_source_provider: Optional[LocalSourceProvider] = None,
    ):
        self.intelligence_engine = intelligence_engine
        self.planner = planner if planner is not None else ClipPlanner()
        self.renderer = renderer if renderer is not None else FFmpegRenderer()
        self.youtube_source_provider = (
            youtube_source_provider
            if youtube_source_provider is not None
            else YouTubeSourceProvider()
        )
        self.local_source_provider = (
            local_source_provider
            if local_source_provider is not None
            else LocalSourceProvider()
        )

    def run(self, source_location: str) -> list[FinalRenderedClip]:
        workspace, was_existing = Config.get_or_create_workspace(source_location)

        source = self._ensure_source(
            source_location=source_location,
            workspace=workspace,
        )

        state_checker = ArtifactStateChecker(workspace)

        print()
        print("=" * 60)
        print("PIPELINE")
        print("=" * 60)
        print(f"Project: {workspace.project_id}")

        if was_existing:
            print("Existing project detected. Checking checkpoints...")

        transcript = self._ensure_transcript(
            source=source,
            workspace=workspace,
            state_checker=state_checker,
        )

        candidate_manifest = self._ensure_candidates(
            transcript=transcript,
            workspace=workspace,
            state_checker=state_checker,
        )

        evaluation_manifest = self._ensure_evaluations(
            candidate_manifest=candidate_manifest,
            workspace=workspace,
            state_checker=state_checker,
        )

        clip_manifest = self._ensure_clip_plan(
            candidate_manifest=candidate_manifest,
            evaluation_manifest=evaluation_manifest,
            workspace=workspace,
            state_checker=state_checker,
        )

        return self._ensure_rendered_clips(
            source=source,
            clip_manifest=clip_manifest,
            workspace=workspace,
        )

    def _ensure_source(
        self,
        source_location: str,
        workspace: ProjectWorkspace,
    ) -> VideoSource:
        source_path = workspace.source_dir / "source.json"

        if source_path.exists():
            return load_source(workspace)

        source_type = (
            "youtube"
            if Config.is_youtube_url(source_location)
            else "local"
        )

        source = VideoSource(
            source_type=source_type,
            location=source_location,
        )

        save_source(source, workspace)

        return source

    def _ensure_transcript(
        self,
        source: VideoSource,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        from src.ingestion.downloader import ingest_audio
        from src.perception.whisper_engine import transcribe_audio

        state = state_checker.get_state()

        if state.transcript_exists:
            print("Transcript checkpoint found. Loading...")
            return load_transcript(workspace)

        print("Transcription checkpoint missing. Running Whisper...")

        # Transcript is an upstream dependency for every discovery/
        # intelligence/planning/rendering artifact below it.
        self._invalidate_from_candidates_downstream(workspace)

        audio_path = workspace.audio_dir / "audio.mp3"

        if not audio_path.exists():
            print("Ingesting source audio...")

            ingested_source, audio_path = ingest_audio(
                source_location=source.location,
                workspace=workspace,
            )

            source.source_type = ingested_source.source_type
            source.location = ingested_source.location
            source.title = ingested_source.title

            save_source(source, workspace)

        return transcribe_audio(
            audio_path=audio_path,
            source=source,
            workspace=workspace,
        )

    def _ensure_candidates(
        self,
        transcript,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        state = state_checker.get_state()

        if state.candidates_exists:
            print("Candidate checkpoint found. Loading...")
            return load_candidate_manifest(workspace)

        print("Candidate checkpoint missing. Generating candidates...")

        # Candidates are upstream of evaluations, clip planning,
        # source sections, and final rendered clips.
        self._invalidate_from_evaluations_downstream(workspace)

        return generate_candidate_windows(
            transcript=transcript,
            workspace=workspace,
        )

    def _ensure_evaluations(
        self,
        candidate_manifest,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        state = state_checker.get_state()

        if state.evaluations_exists:
            print("Evaluation checkpoint found. Loading...")
            return load_evaluation_manifest(workspace)

        print("Evaluation checkpoint missing. Running Gemini...")

        # A regenerated evaluation can change which clips should be
        # selected. Therefore the old plan and everything derived from
        # that plan must not survive.
        self._invalidate_from_clip_plan_downstream(workspace)

        evaluation_manifest = self.intelligence_engine.evaluate_candidates(
            candidate_manifest
        )

        save_evaluation_manifest(
            evaluation_manifest,
            workspace,
        )

        return evaluation_manifest

    def _ensure_clip_plan(
        self,
        candidate_manifest,
        evaluation_manifest,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        state = state_checker.get_state()

        if state.clip_plan_exists:
            print("Clip-plan checkpoint found. Loading...")
            return load_clip_manifest(workspace)

        print("Clip-plan checkpoint missing. Planning clips...")

        # A regenerated clip plan changes the exact source ranges that
        # must be acquired and rendered.
        self._invalidate_render_artifacts(workspace)

        clip_manifest = self.planner.plan(
            candidate_manifest=candidate_manifest,
            evaluation_manifest=evaluation_manifest,
        )

        save_clip_manifest(
            clip_manifest,
            workspace,
        )

        return clip_manifest

    def _ensure_rendered_clips(
        self,
        source: VideoSource,
        clip_manifest,
        workspace: ProjectWorkspace,
    ):
        rendered_clips = []

        for decision in clip_manifest.selected_clips:
            title = decision.title or f"Clip {decision.clip_id}"

            safe_title = ArtifactStateChecker._safe_filename(title)

            output_path = (
                workspace.clips_dir
                / f"clip_{decision.clip_id:02d}_{safe_title}.mp4"
            )

            if output_path.exists():
                print()
                print(
                    f"Existing rendered clip "
                    f"{decision.clip_id} found."
                )
                print("Validating existing output...")

                try:
                    self.renderer.validate_media(output_path)

                except RenderingError:
                    print(
                        f"Existing clip {decision.clip_id} "
                        "failed validation."
                    )
                    print(
                        "Removing invalid output and re-rendering..."
                    )

                    if output_path.exists():
                        output_path.unlink()

                else:
                    print(
                        f"Rendered clip {decision.clip_id} "
                        "is valid. Skipping."
                    )

                    rendered_clips.append(
                        FinalRenderedClip(
                            clip_id=decision.clip_id,
                            file_path=str(output_path),
                            title=title,
                        )
                    )

                    continue

            acquired_path = (
                workspace.source_dir
                / f"section_{decision.clip_id:02d}.mp4"
            )

            # A source section is derived from the current clip plan.
            # If it exists, validate it before trusting it.
            if acquired_path.exists():
                print()
                print(
                    f"Existing source section for clip "
                    f"{decision.clip_id} found."
                )
                print("Validating existing source section...")

                try:
                    self.youtube_source_provider.validate_media(
                        acquired_path
                    )
                except Exception:
                    try:
                        self.local_source_provider.validate_media(
                            acquired_path
                        )
                    except Exception:
                        print(
                            "Existing source section failed validation."
                        )
                        print(
                            "Removing invalid source section "
                            "and reacquiring..."
                        )

                        if acquired_path.exists():
                            acquired_path.unlink()

            if not acquired_path.exists():
                print()
                print(
                    f"Acquiring source section for clip "
                    f"{decision.clip_id}..."
                )
                print(
                    f"Range: "
                    f"{decision.snapped_start_time:.2f}s → "
                    f"{decision.snapped_end_time:.2f}s"
                )

                try:
                    if source.source_type == "youtube":
                        source_path = (
                            self.youtube_source_provider.acquire_section(
                                source_url=source.location,
                                output_path=acquired_path,
                                start_time=decision.snapped_start_time,
                                end_time=decision.snapped_end_time,
                            )
                        )

                    elif source.source_type == "local":
                        source_path = (
                            self.local_source_provider.acquire_section(
                                source_path=Path(source.location),
                                output_path=acquired_path,
                                start_time=decision.snapped_start_time,
                                end_time=decision.snapped_end_time,
                            )
                        )

                    else:
                        raise PipelineError(
                            f"Unsupported source type: "
                            f"{source.source_type}"
                        )

                except SourceAcquisitionError as exc:
                    raise PipelineError(
                        f"Could not acquire source section for "
                        f"clip {decision.clip_id}:\n{exc}"
                    ) from exc

            else:
                source_path = acquired_path

            print(f"Rendering clip {decision.clip_id}...")

            try:
                rendered_path = self.renderer.render_clip(
                    source_path=source_path,
                    output_path=output_path,
                    start_time=0.0,
                    end_time=decision.duration,
                )

            except RenderingError as exc:
                raise PipelineError(
                    f"Could not render clip {decision.clip_id}:\n{exc}"
                ) from exc

            rendered_clips.append(
                FinalRenderedClip(
                    clip_id=decision.clip_id,
                    file_path=str(rendered_path),
                    title=title,
                )
            )

        return rendered_clips

    # ------------------------------------------------------------------
    # Artifact invalidation
    # ------------------------------------------------------------------

    @staticmethod
    def _delete_file(path: Path) -> None:
        """Delete a single artifact if it exists."""
        if path.exists():
            path.unlink()

    @classmethod
    def _invalidate_from_candidates_downstream(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Invalidate everything downstream of the transcript.

        Dependency chain:

            transcript
                ↓
            candidates
                ↓
            evaluations
                ↓
            clip plan
                ↓
            source sections
                ↓
            rendered clips
        """

        cls._invalidate_from_evaluations_downstream(workspace)

        cls._delete_file(
            workspace.candidates_dir / "candidates.json"
        )

    @classmethod
    def _invalidate_from_evaluations_downstream(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Invalidate everything downstream of candidate generation.

        Candidates can change which evaluations are valid, which can
        change the clip plan and therefore all acquired/rendered media.
        """

        cls._invalidate_from_clip_plan_downstream(workspace)

        cls._delete_file(
            workspace.evaluations_dir / "evaluations.json"
        )

    @classmethod
    def _invalidate_from_clip_plan_downstream(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Invalidate everything derived from the clip plan.

        If the evaluation changes, the selected clips may change.
        Therefore the old plan, source sections, and final renders
        cannot be trusted.
        """

        cls._delete_file(
            workspace.evaluations_dir / "clip_plan.json"
        )

        cls._invalidate_render_artifacts(workspace)

    @classmethod
    def _invalidate_render_artifacts(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Remove media derived from the current clip plan.

        Source sections and final rendered clips are both disposable
        derived artifacts. They will be regenerated from the new plan.
        """

        if workspace.source_dir.exists():
            for section_path in workspace.source_dir.glob(
                "section_*.mp4"
            ):
                cls._delete_file(section_path)

        if workspace.clips_dir.exists():
            for clip_path in workspace.clips_dir.glob(
                "*.mp4"
            ):
                cls._delete_file(clip_path)