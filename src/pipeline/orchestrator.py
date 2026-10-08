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

from src.editing.planning.editing_planner import EditingPlanner
from src.editing.rendering.editing_renderer import (
    EditingRenderer,
    EditingRenderingError,
)

from src.packaging.base import BasePackager
from src.packaging.gemini_packager import (
    GeminiPackager,
    PackagingError,
)
from src.packaging.models import ClipPackaging
from src.packaging.persistence import (
    load_clip_packaging,
    packaging_path,
    save_clip_packaging,
)
from src.thumbnails.domain.results import ThumbnailResultStatus
from src.thumbnails.pipeline_stage import ThumbnailGenerator


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
        packager: Optional[BasePackager] = None,
        editing_planner: Optional[EditingPlanner] = None,
        editing_renderer: Optional[EditingRenderer] = None,
        thumbnail_stage: Optional[ThumbnailGenerator] = None,
    ):
        self.intelligence_engine = intelligence_engine

        self.planner = (
            planner
            if planner is not None
            else ClipPlanner()
        )

        self.renderer = (
            renderer
            if renderer is not None
            else FFmpegRenderer()
        )

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

        self.packager = (
            packager
            if packager is not None
            else GeminiPackager()
        )

        self.editing_planner = (
            editing_planner
            if editing_planner is not None
            else EditingPlanner()
        )

        self.editing_renderer = (
            editing_renderer
            if editing_renderer is not None
            else EditingRenderer()
        )
        # Optional for library callers and tests; the primary CLI wires V2 in.
        self.thumbnail_stage = thumbnail_stage

    def run(
        self,
        source_location: str,
    ) -> list[FinalRenderedClip]:
        workspace, was_existing = (
            Config.get_or_create_workspace(
                source_location
            )
        )

        source = self._ensure_source(
            source_location=source_location,
            workspace=workspace,
        )

        state_checker = ArtifactStateChecker(
            workspace
        )

        print()
        print("=" * 60)
        print("PIPELINE")
        print("=" * 60)
        print(
            f"Project: {workspace.project_id}"
        )

        if was_existing:
            print(
                "Existing project detected. "
                "Checking checkpoints..."
            )

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

        return self._ensure_rendered_and_packaged_clips(
            source=source,
            transcript=transcript,
            candidate_manifest=candidate_manifest,
            clip_manifest=clip_manifest,
            workspace=workspace,
        )

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def _ensure_source(
        self,
        source_location: str,
        workspace: ProjectWorkspace,
    ) -> VideoSource:
        source_path = (
            workspace.source_dir / "source.json"
        )

        if source_path.exists():
            return load_source(workspace)

        source_type = (
            "youtube"
            if Config.is_youtube_url(
                source_location
            )
            else "local"
        )

        source = VideoSource(
            source_type=source_type,
            location=source_location,
        )

        save_source(
            source,
            workspace,
        )

        return source

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def _ensure_transcript(
        self,
        source: VideoSource,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        from src.ingestion.downloader import ingest_audio
        from src.perception.whisper_engine import (
            transcribe_audio,
        )

        state = state_checker.get_state()

        if state.transcript_exists:
            print(
                "Transcript checkpoint found. Loading..."
            )

            return load_transcript(workspace)

        print(
            "Transcription checkpoint missing. "
            "Running Whisper..."
        )

        self._invalidate_from_candidates_downstream(
            workspace
        )

        audio_path = (
            workspace.audio_dir / "audio.mp3"
        )

        if not audio_path.exists():
            print(
                "Ingesting source audio..."
            )

            ingested_source, audio_path = (
                ingest_audio(
                    source_location=source.location,
                    workspace=workspace,
                )
            )

            source.source_type = (
                ingested_source.source_type
            )
            source.location = (
                ingested_source.location
            )
            source.title = (
                ingested_source.title
            )

            save_source(
                source,
                workspace,
            )

        return transcribe_audio(
            audio_path=audio_path,
            source=source,
            workspace=workspace,
        )

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def _ensure_candidates(
        self,
        transcript,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        state = state_checker.get_state()

        if state.candidates_exists:
            print(
                "Candidate checkpoint found. Loading..."
            )

            return load_candidate_manifest(
                workspace
            )

        print(
            "Candidate checkpoint missing. "
            "Generating candidates..."
        )

        self._invalidate_from_evaluations_downstream(
            workspace
        )

        return generate_candidate_windows(
            transcript=transcript,
            workspace=workspace,
        )

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def _ensure_evaluations(
        self,
        candidate_manifest,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        state = state_checker.get_state()

        if state.evaluations_exists:
            print(
                "Evaluation checkpoint found. Loading..."
            )

            evaluation_manifest = load_evaluation_manifest(
                workspace
            )

            # Boundary refinement was added after older evaluation artifacts
            # were created. Force one Gemini pass when the checkpoint lacks
            # the new segment-level boundaries.
            if all(
                evaluation.start_segment_id is not None
                and evaluation.end_segment_id is not None
                for evaluation in evaluation_manifest.evaluations
            ):
                return evaluation_manifest

            print(
                "Evaluation checkpoint uses the legacy boundary format. "
                "Regenerating with AI boundary refinement..."
            )
            self._invalidate_from_clip_plan_downstream(
                workspace
            )

        print(
            "Evaluation checkpoint missing. "
            "Running Gemini..."
        )

        self._invalidate_from_clip_plan_downstream(
            workspace
        )

        evaluation_manifest = (
            self.intelligence_engine.evaluate_candidates(
                candidate_manifest
            )
        )

        save_evaluation_manifest(
            evaluation_manifest,
            workspace,
        )

        return evaluation_manifest

    # ------------------------------------------------------------------
    # Clip planning
    # ------------------------------------------------------------------

    def _ensure_clip_plan(
        self,
        candidate_manifest,
        evaluation_manifest,
        workspace: ProjectWorkspace,
        state_checker: ArtifactStateChecker,
    ):
        state = state_checker.get_state()

        if state.clip_plan_exists:
            print(
                "Clip-plan checkpoint found. Loading..."
            )

            return load_clip_manifest(
                workspace
            )

        print(
            "Clip-plan checkpoint missing. "
            "Planning clips..."
        )

        self._invalidate_render_artifacts(
            workspace
        )

        clip_manifest = self.planner.plan(
            candidate_manifest=candidate_manifest,
            evaluation_manifest=evaluation_manifest,
        )

        save_clip_manifest(
            clip_manifest,
            workspace,
        )

        return clip_manifest

    # ------------------------------------------------------------------
    # Rendering + Editing + Packaging
    # ------------------------------------------------------------------

    def _ensure_rendered_and_packaged_clips(
        self,
        source: VideoSource,
        transcript,
        candidate_manifest,
        clip_manifest,
        workspace: ProjectWorkspace,
    ) -> list[FinalRenderedClip]:
        rendered_clips = []

        for decision in clip_manifest.selected_clips:
            title = (
                decision.title
                or f"Clip {decision.clip_id}"
            )

            state_checker = ArtifactStateChecker(
                workspace
            )

            final_output_path = (
                state_checker.final_clip_path(
                    clip_id=decision.clip_id,
                    title=title,
                )
            )

            final_output_path = (
                self._ensure_edited_clip(
                    source=source,
                    transcript=transcript,
                    decision=decision,
                    output_path=final_output_path,
                    workspace=workspace,
                )
            )

            rendered_clips.append(
                FinalRenderedClip(
                    clip_id=decision.clip_id,
                    file_path=str(
                        final_output_path
                    ),
                    title=title,
                )
            )

            packaging = self._ensure_packaging(
                decision=decision,
                candidate_manifest=candidate_manifest,
                source=source,
                workspace=workspace,
            )

            if self.thumbnail_stage is not None:
                self._ensure_thumbnail(
                    source=source,
                    decision=decision,
                    packaging=packaging,
                    workspace=workspace,
                )

        return rendered_clips

    def _ensure_edited_clip(
        self,
        source: VideoSource,
        transcript,
        decision,
        output_path: Path,
        workspace: ProjectWorkspace,
    ) -> Path:
        state_checker = ArtifactStateChecker(
            workspace
        )

        base_render_path = (
            state_checker.base_render_path(
                decision.clip_id
            )
        )

        if output_path.exists():
            print()
            print(
                f"Existing final clip "
                f"{decision.clip_id} found."
            )

            print(
                "Validating existing final output..."
            )

            final_valid = False

            try:
                self.editing_renderer.validate_media(
                    output_path
                )
            except EditingRenderingError:
                print(
                    f"Existing final clip "
                    f"{decision.clip_id} "
                    "failed validation."
                )
            else:
                final_valid = True

            if final_valid and base_render_path.exists():
                try:
                    self.renderer.validate_media(
                        base_render_path
                    )
                except RenderingError:
                    print(
                        f"Base render for clip "
                        f"{decision.clip_id} "
                        "failed validation."
                    )

                    print(
                        "The final clip will be rebuilt "
                        "from a fresh base render."
                    )
                else:
                    print(
                        f"Final clip "
                        f"{decision.clip_id} "
                        "is valid. Skipping editing."
                    )

                    return output_path

            print(
                f"Removing stale final clip "
                f"{decision.clip_id}..."
            )

            self._delete_file(
                output_path
            )

        base_render_path = (
            self._ensure_base_render(
                source=source,
                decision=decision,
                output_path=base_render_path,
                workspace=workspace,
            )
        )

        print()
        print(
            f"Planning edits for clip "
            f"{decision.clip_id}..."
        )

        try:
            editing_plan = (
                self.editing_planner.create_plan(
                    clip_id=(
                        f"clip_{decision.clip_id:02d}"
                    ),
                    transcript=transcript,
                    clip_start_time=(
                        decision.snapped_start_time
                    ),
                    clip_end_time=(
                        decision.snapped_end_time
                    ),
                    content_reason=decision.reason,
                    title=decision.title,
                )
            )

        except (TypeError, ValueError) as exc:
            raise PipelineError(
                f"Could not create editing plan "
                f"for clip "
                f"{decision.clip_id}:\n{exc}"
            ) from exc

        print()
        print(
            f"Rendering edited clip "
            f"{decision.clip_id}..."
        )

        try:
            return self.editing_renderer.render(
                source_path=base_render_path,
                output_path=output_path,
                plan=editing_plan,
            )

        except EditingRenderingError as exc:
            raise PipelineError(
                f"Could not render edited clip "
                f"{decision.clip_id}:\n{exc}"
            ) from exc

    def _ensure_base_render(
        self,
        source: VideoSource,
        decision,
        output_path: Path,
        workspace: ProjectWorkspace,
    ) -> Path:
        if output_path.exists():
            print()
            print(
                f"Existing base render for clip "
                f"{decision.clip_id} found."
            )

            print(
                "Validating existing base render..."
            )

            try:
                self.renderer.validate_media(
                    output_path
                )

            except RenderingError:
                print(
                    f"Existing base render "
                    f"{decision.clip_id} "
                    "failed validation."
                )

                print(
                    "Removing invalid base render "
                    "and rebuilding..."
                )

                self._delete_file(
                    output_path
                )

            else:
                print(
                    f"Base render "
                    f"{decision.clip_id} "
                    "is valid. Skipping base render."
                )

                return output_path

        acquired_path = (
            workspace.source_dir
            / (
                f"section_"
                f"{decision.clip_id:02d}.mp4"
            )
        )

        acquired_path = (
            self._ensure_source_section(
                source=source,
                decision=decision,
                acquired_path=acquired_path,
            )
        )

        print(
            f"Rendering base clip "
            f"{decision.clip_id}..."
        )

        try:
            rendered_path = (
                self.renderer.render_clip(
                    source_path=acquired_path,
                    output_path=output_path,
                    start_time=0.0,
                    end_time=decision.duration,
                )
            )

        except RenderingError as exc:
            raise PipelineError(
                f"Could not render base clip "
                f"{decision.clip_id}:\n{exc}"
            ) from exc

        return rendered_path

    def _ensure_source_section(
        self,
        source: VideoSource,
        decision,
        acquired_path: Path,
    ) -> Path:
        if acquired_path.exists():
            print()
            print(
                f"Existing source section for clip "
                f"{decision.clip_id} found."
            )

            print(
                "Validating existing source section..."
            )

            if self._validate_source_section(
                acquired_path
            ):
                print(
                    "Source section is valid. "
                    "Skipping acquisition."
                )

                return acquired_path

            print(
                "Existing source section failed "
                "validation."
            )

            print(
                "Removing invalid source section "
                "and reacquiring..."
            )

            self._delete_file(
                acquired_path
            )

        print()
        print(
            f"Acquiring source section for clip "
            f"{decision.clip_id}..."
        )

        print(
            f"Range: "
            f"{decision.snapped_start_time:.2f}s "
            f"-> "
            f"{decision.snapped_end_time:.2f}s"
        )

        try:
            if source.source_type == "youtube":
                source_path = (
                    self.youtube_source_provider.acquire_section(
                        source_url=source.location,
                        output_path=acquired_path,
                        start_time=(
                            decision.snapped_start_time
                        ),
                        end_time=(
                            decision.snapped_end_time
                        ),
                    )
                )

            elif source.source_type == "local":
                source_path = (
                    self.local_source_provider.acquire_section(
                        source_path=Path(
                            source.location
                        ),
                        output_path=acquired_path,
                        start_time=(
                            decision.snapped_start_time
                        ),
                        end_time=(
                            decision.snapped_end_time
                        ),
                    )
                )

            else:
                raise PipelineError(
                    f"Unsupported source type: "
                    f"{source.source_type}"
                )

        except SourceAcquisitionError as exc:
            raise PipelineError(
                f"Could not acquire source section "
                f"for clip "
                f"{decision.clip_id}:\n{exc}"
            ) from exc

        return source_path

    def _validate_source_section(
        self,
        acquired_path: Path,
    ) -> bool:
        try:
            self.youtube_source_provider.validate_media(
                acquired_path
            )

            return True

        except Exception:
            pass

        try:
            self.local_source_provider.validate_media(
                acquired_path
            )

            return True

        except Exception:
            return False

    # ------------------------------------------------------------------
    # Packaging
    # ------------------------------------------------------------------

    def _ensure_packaging(
        self,
        decision,
        candidate_manifest,
        source: VideoSource,
        workspace: ProjectWorkspace,
    ) -> ClipPackaging:
        path = packaging_path(
            workspace,
            decision.clip_id,
        )

        if path.exists():
            print()
            print(
                f"Existing packaging for clip "
                f"{decision.clip_id} found."
            )

            print(
                "Validating packaging checkpoint..."
            )

            try:
                packaging = load_clip_packaging(
                    workspace,
                    decision.clip_id,
                )

            except Exception:
                print(
                    "Packaging checkpoint is invalid."
                )

                print(
                    "Removing invalid packaging "
                    "and regenerating..."
                )

                self._delete_file(path)

            else:
                print(
                    f"Packaging for clip "
                    f"{decision.clip_id} "
                    "is valid. Skipping."
                )

                return packaging

        transcript_text = (
            self._get_candidate_transcript(
                decision=decision,
                candidate_manifest=candidate_manifest,
            )
        )

        source_title = (
            source.title
            or source.location
        )

        print()
        print(
            f"Packaging clip "
            f"{decision.clip_id} with Gemini..."
        )

        # Packaging is the source of the thumbnail's creative copy. If it must
        # be regenerated, any thumbnail derived from the old packaging is stale.
        self._delete_file(
            workspace.root_dir
            / "thumbnails"
            / f"clip_{decision.clip_id:02d}_thumbnail.jpg"
        )

        try:
            data = self.packager.package_clip(
                clip=decision,
                transcript_text=transcript_text,
                source_title=source_title,
            )

            packaging = ClipPackaging(
                clip_id=decision.clip_id,
                title=str(data["title"]),
                hook=str(data["hook"]),
                caption=str(data["caption"]),
                description=str(
                    data["description"]
                ),
                thumbnail_text=str(
                    data["thumbnail_text"]
                ),
                content_angle=str(
                    data["content_angle"]
                ),
                hashtags=[
                    str(hashtag)
                    for hashtag in data.get(
                        "hashtags",
                        [],
                    )
                ],
            )

            save_clip_packaging(
                packaging,
                workspace,
            )

        except PackagingError as exc:
            raise PipelineError(
                f"Could not package clip "
                f"{decision.clip_id}:\n{exc}"
            ) from exc

        except KeyError as exc:
            raise PipelineError(
                f"Packaging response for clip "
                f"{decision.clip_id} "
                f"is missing required field: "
                f"{exc}"
            ) from exc

        except (TypeError, ValueError) as exc:
            raise PipelineError(
                f"Packaging response for clip "
                f"{decision.clip_id} "
                f"contains invalid data:\n{exc}"
            ) from exc

        print(
            f"Packaging saved for clip "
            f"{decision.clip_id}."
        )

        return packaging

    # ------------------------------------------------------------------
    # Thumbnail V2 (local; reuses existing packaging)
    # ------------------------------------------------------------------

    def _ensure_thumbnail(
        self,
        source: VideoSource,
        decision,
        packaging: ClipPackaging,
        workspace: ProjectWorkspace,
    ) -> None:
        """Generate a thumbnail without making another AI request.

        The image is a checkpoint. A new packaging artifact invalidates its
        thumbnail; a valid existing image is reused on resume.
        """
        output_path = (
            workspace.root_dir
            / "thumbnails"
            / f"clip_{decision.clip_id:02d}_thumbnail.jpg"
        )
        candidates_dir = (
            workspace.root_dir
            / "thumbnail_work"
            / f"clip_{decision.clip_id:02d}"
        )

        if output_path.is_file():
            try:
                from PIL import Image

                with Image.open(output_path) as image:
                    image.verify()
                    if image.size != (1080, 1920):
                        raise ValueError(
                            "thumbnail checkpoint has incorrect dimensions "
                            f"{image.size}; expected (1080, 1920) for Shorts."
                        )
            except (OSError, ValueError):
                print(
                    f"Existing thumbnail for clip {decision.clip_id} "
                    "is invalid; regenerating."
                )
                self._delete_file(output_path)
            else:
                print(
                    f"Thumbnail checkpoint for clip {decision.clip_id} "
                    "is valid. Skipping."
                )
                return

        source_section = workspace.source_dir / (
            f"section_{decision.clip_id:02d}.mp4"
        )
        try:
            source_section = self._ensure_source_section(
                source=source,
                decision=decision,
                acquired_path=source_section,
            )
            result = self.thumbnail_stage.generate(
                source_video_path=source_section,
                packaging=packaging,
                output_path=output_path,
                candidates_dir=candidates_dir,
            )
            if (
                result.status in {
                    ThumbnailResultStatus.SUCCESS,
                    ThumbnailResultStatus.FALLBACK_SUCCESS,
                }
                and result.output_path
                and Path(result.output_path).is_file()
            ):
                print(
                    f"Thumbnail generated for clip {decision.clip_id}: "
                    f"{result.output_path}"
                )
            else:
                print(
                    f"Thumbnail skipped for clip {decision.clip_id}: "
                    f"{result.failure_reason or result.status.value}"
                )
        except Exception as exc:
            # Thumbnail generation must not discard an otherwise valid clip.
            print(
                f"Thumbnail generation failed for clip "
                f"{decision.clip_id}: {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _get_candidate_transcript(
        decision,
        candidate_manifest,
    ) -> str:
        for candidate in (
            candidate_manifest.candidates
        ):
            if (
                candidate.candidate_id
                == decision.candidate_id
            ):
                transcript_text = (
                    candidate.transcript_text
                )

                if not transcript_text.strip():
                    raise PipelineError(
                        f"Candidate "
                        f"{decision.candidate_id} "
                        f"has no transcript text. "
                        f"Cannot package clip "
                        f"{decision.clip_id}."
                    )

                return transcript_text

        raise PipelineError(
            f"Could not find candidate "
            f"{decision.candidate_id} "
            f"for clip "
            f"{decision.clip_id}. "
            "Packaging cannot continue."
        )

    # ------------------------------------------------------------------
    # Artifact invalidation
    # ------------------------------------------------------------------

    @staticmethod
    def _delete_file(
        path: Path,
    ) -> None:
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
            base renders
                ↓
            edited final clips
                ↓
            packaging
        """

        cls._invalidate_from_evaluations_downstream(
            workspace
        )

        cls._delete_file(
            workspace.candidates_dir
            / "candidates.json"
        )

    @classmethod
    def _invalidate_from_evaluations_downstream(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Invalidate everything downstream of
        candidate generation.

        Candidates can change which evaluations
        are valid, which can change the clip plan,
        rendering, editing, and packaging.
        """

        cls._invalidate_from_clip_plan_downstream(
            workspace
        )

        cls._delete_file(
            workspace.evaluations_dir
            / "evaluations.json"
        )

    @classmethod
    def _invalidate_from_clip_plan_downstream(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Invalidate everything derived from the
        clip plan.

        If the evaluation changes, the selected
        clips may change. Therefore the old plan,
        source sections, base renders, edited clips,
        and packaging cannot be trusted.
        """

        cls._delete_file(
            workspace.evaluations_dir
            / "clip_plan.json"
        )

        cls._invalidate_render_artifacts(
            workspace
        )

    @classmethod
    def _invalidate_render_artifacts(
        cls,
        workspace: ProjectWorkspace,
    ) -> None:
        """
        Remove media and packaging derived from
        the current clip plan.
        """

        if workspace.source_dir.exists():
            for section_path in (
                workspace.source_dir.glob(
                    "section_*.mp4"
                )
            ):
                cls._delete_file(
                    section_path
                )

        if workspace.renders_dir.exists():
            for render_path in (
                workspace.renders_dir.glob(
                    "*.mp4"
                )
            ):
                cls._delete_file(
                    render_path
                )

        if workspace.clips_dir.exists():
            for clip_path in (
                workspace.clips_dir.glob(
                    "*.mp4"
                )
            ):
                cls._delete_file(
                    clip_path
                )

        thumbnails_dir = workspace.root_dir / "thumbnails"
        if thumbnails_dir.exists():
            for thumbnail_path in thumbnails_dir.glob("clip_*_thumbnail.jpg"):
                cls._delete_file(thumbnail_path)

        if workspace.packaging_dir.exists():
            for packaging_path_item in (
                workspace.packaging_dir.glob(
                    "clip_*.json"
                )
            ):
                cls._delete_file(
                    packaging_path_item
                )