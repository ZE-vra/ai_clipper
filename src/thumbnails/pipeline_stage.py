"""Local V2 thumbnail stage that reuses clip packaging and makes no AI calls."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.packaging.models import ClipPackaging
from src.thumbnails.domain.geometry import Size
from src.thumbnails.domain.results import ThumbnailResult
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.frame_extractor import FFmpegFrameExtractor
from src.thumbnails.intelligence.packaging_adapter import (
    thumbnail_inputs_from_packaging,
)
from src.thumbnails.orchestration.thumbnail_orchestrator import (
    ThumbnailOrchestrator,
)
from src.thumbnails.perception.frame_discovery import FrameDiscovery
from src.thumbnails.perception.frame_perception_analyzer import (
    FramePerceptionAnalyzer,
)
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence
from src.thumbnails.perception.video_frame_sampler import (
    FFprobeVideoDurationReader,
    ThumbnailFrameSampler,
)


class _NoSubjectModel:
    """Explicit no-model mode; do not download or load segmentation weights."""

    def analyze(self, image_path: Path) -> SubjectAnalysisEvidence:
        return SubjectAnalysisEvidence()


class ThumbnailGenerator(Protocol):
    """Pipeline-facing contract for a deterministic, checkpointable thumbnail stage."""

    def generate(
        self,
        *,
        source_video_path: str | Path,
        packaging: ClipPackaging,
        output_path: str | Path,
        candidates_dir: str | Path,
    ) -> ThumbnailResult:
        ...


class ThumbnailV2Stage:
    """Generate a thumbnail from an already-selected clip's clean source section.

    All AI-derived copy comes from the existing ClipPackaging artifact. Frame
    extraction, perception, composition, rendering, and evaluation are local.
    """

    def __init__(
        self,
        *,
        orchestrator: ThumbnailOrchestrator | None = None,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        sample_count: int = 9,
    ) -> None:
        if sample_count < 1:
            raise ValueError("sample_count must be at least 1.")

        self._orchestrator = orchestrator or self._build_orchestrator(
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
            sample_count=sample_count,
        )

    @staticmethod
    def _build_orchestrator(
        *,
        ffmpeg_binary: str,
        ffprobe_binary: str,
        sample_count: int,
    ) -> ThumbnailOrchestrator:
        sampler = ThumbnailFrameSampler(
            extractor=FFmpegFrameExtractor(ffmpeg_binary=ffmpeg_binary),
            duration_reader=FFprobeVideoDurationReader(
                ffprobe_binary=ffprobe_binary
            ).read,
            sample_count=sample_count,
        )
        discovery = FrameDiscovery(
            sampler=sampler,
            perception=FramePerceptionAnalyzer(
                subject_analyzer=_NoSubjectModel()
            ),
        )
        return ThumbnailOrchestrator(frame_discovery=discovery)

    def generate(
        self,
        *,
        source_video_path: str | Path,
        packaging: ClipPackaging,
        output_path: str | Path,
        candidates_dir: str | Path,
    ) -> ThumbnailResult:
        brief, understanding = thumbnail_inputs_from_packaging(packaging)
        return self._orchestrator.generate(
            source_video_path=source_video_path,
            output_path=output_path,
            candidates_dir=candidates_dir,
            brief=brief,
            understanding=understanding,
            target=ThumbnailTarget(
                target_id="youtube-16x9",
                platform="youtube",
                size=Size(width=1280, height=720),
            ),
        )
