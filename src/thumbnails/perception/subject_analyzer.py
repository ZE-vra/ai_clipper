from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.thumbnails.perception.subjects import SubjectAnalysisEvidence


class SubjectAnalyzer(Protocol):
    """
    Contract for semantic subject detection in a frame.

    Implementations detect subjects such as people and faces.
    They return perception evidence only; they do not make
    thumbnail-selection or composition decisions.
    """

    def analyze(
        self,
        image_path: Path,
    ) -> SubjectAnalysisEvidence:
        """
        Analyze one frame and return detected subject evidence.
        """
        ...