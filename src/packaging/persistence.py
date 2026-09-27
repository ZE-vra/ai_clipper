"""Persistence layer for packaging artifacts."""

import json
from pathlib import Path
from typing import Any, Dict

from src.config import ProjectWorkspace
from src.exceptions import ClipperError
from src.packaging.models import ClipPackaging


class PackagingPersistenceError(ClipperError):
    """Raised when packaging data cannot be saved or loaded."""


def _save_json(path: Path, data: Dict[str, Any]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except OSError as exc:
        raise PackagingPersistenceError(
            f"Could not save packaging artifact: {path}"
        ) from exc

    return path


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise PackagingPersistenceError(
            f"Packaging artifact does not exist: {path}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError) as exc:
        raise PackagingPersistenceError(
            f"Could not load packaging artifact: {path}"
        ) from exc


def packaging_path(
    workspace: ProjectWorkspace,
    clip_id: int,
) -> Path:
    """Return the checkpoint path for a clip's packaging."""

    return (
        workspace.packaging_dir
        / f"clip_{clip_id:02d}.json"
    )


def save_clip_packaging(
    packaging: ClipPackaging,
    workspace: ProjectWorkspace,
) -> Path:
    """Save packaging metadata for one clip."""

    path = packaging_path(
        workspace,
        packaging.clip_id,
    )

    return _save_json(
        path,
        packaging.to_dict(),
    )


def load_clip_packaging(
    workspace: ProjectWorkspace,
    clip_id: int,
) -> ClipPackaging:
    """Load packaging metadata for one clip."""

    path = packaging_path(
        workspace,
        clip_id,
    )

    data = _load_json(path)

    try:
        return ClipPackaging(
            clip_id=int(data["clip_id"]),
            title=str(data["title"]),
            hook=str(data["hook"]),
            caption=str(data["caption"]),
            description=str(data["description"]),
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

    except (KeyError, TypeError, ValueError) as exc:
        raise PackagingPersistenceError(
            f"Invalid packaging artifact: {path}"
        ) from exc