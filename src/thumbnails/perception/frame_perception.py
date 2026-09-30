from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.thumbnails.perception.focal import FocalAnalysisEvidence
from src.thumbnails.perception.quality import FrameQualityEvidence
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence


@dataclass(frozen=True)
class FramePerception:
    """
    Complete perception evidence for one frame.

    This record combines independent perception outputs.
    It does not select the frame, choose a thumbnail strategy,
    or make composition decisions.
    """

    frame_path: Path
    quality: FrameQualityEvidence
    focal: FocalAnalysisEvidence
    subjects: SubjectAnalysisEvidence

    def __post_init__(self) -> None:
        if not self.frame_path.name:
            raise ValueError("frame_path must reference a file.")