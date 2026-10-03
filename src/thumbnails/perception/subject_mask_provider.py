from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.thumbnails.domain.geometry import BoundingBox


class SubjectMaskProvider(Protocol):
    """Produce a pixel-aligned foreground mask for a detected subject."""

    def create_mask(
        self,
        *,
        image_path: str | Path,
        subject_bounds: BoundingBox,
        output_path: str | Path,
    ) -> Path:
        """Write a grayscale mask aligned to the source image."""
        ...
