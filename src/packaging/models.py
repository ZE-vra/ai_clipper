"""Data models for the packaging layer."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ClipPackaging:
    """Publish-ready metadata generated for one clip."""

    clip_id: int
    title: str
    hook: str
    caption: str
    description: str
    thumbnail_text: str
    content_angle: str
    hashtags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)