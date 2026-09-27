"""Artifact checkpoint detection for resumable pipeline execution."""

from dataclasses import dataclass
from pathlib import Path
import re

from src.config import ProjectWorkspace


@dataclass(frozen=True)
class ArtifactState:
    """Current persisted state of a project pipeline."""

    audio_exists: bool
    transcript_exists: bool
    candidates_exists: bool
    evaluations_exists: bool
    clip_plan_exists: bool

    @property
    def discovery_complete(self) -> bool:
        return (
            self.audio_exists
            and self.transcript_exists
            and self.candidates_exists
        )

    @property
    def intelligence_complete(self) -> bool:
        return self.evaluations_exists

    @property
    def planning_complete(self) -> bool:
        return self.clip_plan_exists


class ArtifactStateChecker:
    """Knows where pipeline checkpoints live."""

    TRANSCRIPT_FILENAME = "transcript.json"
    CANDIDATES_FILENAME = "candidates.json"
    EVALUATIONS_FILENAME = "evaluations.json"
    CLIP_PLAN_FILENAME = "clip_plan.json"

    def __init__(
        self,
        workspace: ProjectWorkspace,
    ):
        self.workspace = workspace

    @property
    def audio_path(self) -> Path:
        return (
            self.workspace.audio_dir
            / "audio.mp3"
        )

    @property
    def transcript_path(self) -> Path:
        return (
            self.workspace.transcript_dir
            / self.TRANSCRIPT_FILENAME
        )

    @property
    def candidates_path(self) -> Path:
        return (
            self.workspace.candidates_dir
            / self.CANDIDATES_FILENAME
        )

    @property
    def evaluations_path(self) -> Path:
        return (
            self.workspace.evaluations_dir
            / self.EVALUATIONS_FILENAME
        )

    @property
    def clip_plan_path(self) -> Path:
        return (
            self.workspace.evaluations_dir
            / self.CLIP_PLAN_FILENAME
        )

    def get_state(self) -> ArtifactState:
        return ArtifactState(
            audio_exists=self.audio_path.exists(),
            transcript_exists=self.transcript_path.exists(),
            candidates_exists=self.candidates_path.exists(),
            evaluations_exists=self.evaluations_path.exists(),
            clip_plan_exists=self.clip_plan_path.exists(),
        )

    def rendered_clip_path(
        self,
        clip_id: int,
        title: str,
    ) -> Path:
        safe_title = self._safe_filename(
            title
        )

        return (
            self.workspace.clips_dir
            / f"clip_{clip_id:02d}_{safe_title}.mp4"
        )

    @staticmethod
    def _safe_filename(
        value: str,
    ) -> str:
        value = value.strip()

        value = re.sub(
            r'[<>:"/\\|?*\x00-\x1f]',
            "_",
            value,
        )

        value = value.strip(
            " ._-"
        )

        return (
            value[:80]
            or "untitled"
        )