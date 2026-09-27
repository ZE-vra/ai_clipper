"""Configuration and deterministic project workspace management."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from urllib.parse import parse_qs, urlparse


@dataclass
class ProjectWorkspace:
    project_id: str
    root_dir: Path
    source_dir: Path
    audio_dir: Path
    transcript_dir: Path
    candidates_dir: Path
    evaluations_dir: Path
    clips_dir: Path
    logs_dir: Path

    def initialize(self) -> None:
        """Create all directories required by the project."""

        directories = [
            self.root_dir,
            self.source_dir,
            self.audio_dir,
            self.transcript_dir,
            self.candidates_dir,
            self.evaluations_dir,
            self.clips_dir,
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


class Config:
    PROJECTS_ROOT: Path = (
        Path(__file__).resolve().parent.parent / "projects"
    )

    OUTPUT_ROOT: Path = (
        Path(__file__).resolve().parent.parent / "output"
    )

    @staticmethod
    def is_youtube_url(location: str) -> bool:
        """Return True when the location has a recognized YouTube hostname."""

        try:
            parsed = urlparse(location)
        except ValueError:
            return False

        hostname = (
            parsed.hostname or ""
        ).lower()

        return hostname in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        }

    @classmethod
    def get_youtube_video_id(
        cls,
        url: str,
    ) -> str | None:
        """
        Extract a YouTube video ID without making a network request.

        Supports common forms such as:

            https://www.youtube.com/watch?v=VIDEO_ID
            https://youtu.be/VIDEO_ID
            https://www.youtube.com/shorts/VIDEO_ID
            https://www.youtube.com/embed/VIDEO_ID
            https://www.youtube.com/live/VIDEO_ID
        """

        try:
            parsed = urlparse(url)
        except ValueError:
            return None

        hostname = (
            parsed.hostname or ""
        ).lower()

        if hostname in {
            "youtu.be",
            "www.youtu.be",
        }:
            video_id = (
                parsed.path
                .strip("/")
                .split("/")[0]
            )

        elif hostname in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
        }:
            query = parse_qs(
                parsed.query
            )

            if query.get("v"):
                video_id = query["v"][0]

            else:
                path_parts = [
                    part
                    for part in parsed.path.split("/")
                    if part
                ]

                if (
                    len(path_parts) >= 2
                    and path_parts[0] in {
                        "shorts",
                        "embed",
                        "live",
                    }
                ):
                    video_id = path_parts[1]

                else:
                    return None

        else:
            return None

        if not video_id:
            return None

        if not re.fullmatch(
            r"[A-Za-z0-9_-]{6,100}",
            video_id,
        ):
            return None

        return video_id

    @classmethod
    def source_identity(
        cls,
        source_location: str,
    ) -> tuple[str, str]:
        """
        Return:

            (stable_project_id, normalized_source_identity)

        The identity is deterministic, so repeated executions of the
        same source resolve to the same workspace.
        """

        if cls.is_youtube_url(
            source_location
        ):
            video_id = cls.get_youtube_video_id(
                source_location
            )

            if video_id:
                project_id = (
                    f"youtube_{video_id}"
                )

                return (
                    project_id,
                    f"youtube:{video_id}",
                )

            normalized = (
                source_location.strip()
            )

            digest = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()[:16]

            return (
                f"youtube_{digest}",
                f"youtube_url:{normalized}",
            )

        local_path = Path(
            source_location
        ).resolve()

        normalized = str(
            local_path
        ).lower()

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()[:16]

        readable_name = re.sub(
            r"[^\w\-]+",
            "_",
            local_path.stem,
        ).strip("_")

        readable_name = (
            readable_name[-40:]
            or "local_file"
        )

        project_id = (
            f"local_{readable_name}_{digest}"
        )

        return (
            project_id,
            f"local:{normalized}",
        )

    @classmethod
    def workspace_from_project_id(
        cls,
        project_id: str,
    ) -> ProjectWorkspace:
        """Build a workspace object for an existing project ID."""

        project_dir = (
            cls.PROJECTS_ROOT / project_id
        )

        return ProjectWorkspace(
            project_id=project_id,
            root_dir=project_dir,
            source_dir=project_dir / "source",
            audio_dir=project_dir / "audio",
            transcript_dir=project_dir / "transcript",
            candidates_dir=project_dir / "candidates",
            evaluations_dir=project_dir / "evaluations",
            clips_dir=project_dir / "clips",
            logs_dir=project_dir / "logs",
        )

    @classmethod
    def create_workspace(
        cls,
        identifier: str,
    ) -> ProjectWorkspace:
        """
        Create or reuse the deterministic workspace for a source.

        This method is intentionally idempotent.
        """

        project_id, _ = cls.source_identity(
            identifier
        )

        workspace = cls.workspace_from_project_id(
            project_id
        )

        workspace.initialize()

        return workspace

    @classmethod
    def find_workspace(
        cls,
        source_location: str,
    ) -> ProjectWorkspace | None:
        """Return the existing workspace for a source, if one exists."""

        project_id, _ = cls.source_identity(
            source_location
        )

        workspace = cls.workspace_from_project_id(
            project_id
        )

        if not workspace.root_dir.exists():
            return None

        return workspace

    @classmethod
    def workspace_has_pipeline_artifacts(
        cls,
        workspace: ProjectWorkspace,
    ) -> bool:
        """
        Return True when the workspace contains evidence that the
        pipeline has actually started.

        An empty directory is not considered an existing pipeline
        project.
        """

        artifact_paths = [
            workspace.audio_dir / "audio.mp3",
            workspace.transcript_dir / "transcript.json",
            workspace.candidates_dir / "candidates.json",
            workspace.evaluations_dir / "evaluations.json",
            workspace.evaluations_dir / "clip_plan.json",
        ]

        if any(
            path.exists()
            for path in artifact_paths
        ):
            return True

        return any(
            workspace.clips_dir.glob("*.mp4")
        )

    @classmethod
    def get_or_create_workspace(
        cls,
        source_location: str,
    ) -> tuple[ProjectWorkspace, bool]:
        """
        Return:

            (workspace, True)
                when an actual pipeline project already exists.

            (workspace, False)
                when the workspace is new or only an empty directory
                exists.
        """

        existing = cls.find_workspace(
            source_location
        )

        if existing is not None:
            existing.initialize()

            return (
                existing,
                cls.workspace_has_pipeline_artifacts(
                    existing
                ),
            )

        workspace = cls.create_workspace(
            source_location
        )

        return (
            workspace,
            False,
        )