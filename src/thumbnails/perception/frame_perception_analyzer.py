from __future__ import annotations

from pathlib import Path

from src.thumbnails.perception.focal import FocalAnalysisEvidence
from src.thumbnails.perception.focal_analyzer import FocalRegionAnalyzer
from src.thumbnails.perception.frame_perception import FramePerception
from src.thumbnails.perception.quality import FrameQualityEvidence
from src.thumbnails.perception.quality_analyzer import FrameQualityAnalyzer
from src.thumbnails.perception.subject_analyzer import SubjectAnalyzer
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence


class FramePerceptionAnalyzer:
    """
    Coordinates independent perception analyzers for one frame.

    This component combines perception evidence only.
    It does not select frames or make thumbnail decisions.
    """

    def __init__(
        self,
        *,
        quality_analyzer: FrameQualityAnalyzer | None = None,
        focal_analyzer: FocalRegionAnalyzer | None = None,
        subject_analyzer: SubjectAnalyzer | None = None,
    ) -> None:
        self._quality_analyzer = (
            quality_analyzer
            or FrameQualityAnalyzer()
        )
        self._focal_analyzer = (
            focal_analyzer
            or FocalRegionAnalyzer()
        )
        self._subject_analyzer = subject_analyzer

    def analyze(
        self,
        image_path: str | Path,
    ) -> FramePerception:
        path = Path(image_path)

        quality = self._quality_analyzer.analyze(path)
        focal = self._focal_analyzer.analyze(path)

        if self._subject_analyzer is None:
            raise RuntimeError(
                "subject_analyzer must be provided."
            )

        subjects = self._subject_analyzer.analyze(path)

        return FramePerception(
            frame_path=path,
            quality=quality,
            focal=focal,
            subjects=subjects,
        )