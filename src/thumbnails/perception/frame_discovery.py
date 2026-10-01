from __future__ import annotations

from pathlib import Path

from src.thumbnails.domain.assets import (
    AssetProvenance,
    AssetProvenanceKind,
    FrameCandidate,
    VisualAsset,
)
from src.thumbnails.perception.frame_perception_analyzer import FramePerceptionAnalyzer
from src.thumbnails.perception.video_frame_sampler import ThumbnailFrameSampler


class FrameDiscovery:
    """Discovers and perceives candidate frames without selecting a winner."""

    def __init__(
        self,
        *,
        sampler: ThumbnailFrameSampler,
        perception: FramePerceptionAnalyzer,
    ) -> None:
        self.sampler = sampler
        self.perception = perception

    def discover(
        self,
        *,
        video_path: Path,
        output_dir: Path,
    ) -> tuple[FrameCandidate, ...]:
        samples = self.sampler.sample(
            video_path=video_path,
            output_dir=output_dir,
        )

        candidates: list[FrameCandidate] = []
        for index, sample in enumerate(samples, start=1):
            perception = self.perception.analyze(sample.path)
            asset = VisualAsset(
                asset_id=f"frame_{index:03d}_{sample.timestamp:.3f}",
                provenance=AssetProvenance(
                    kind=AssetProvenanceKind.SOURCE_FRAME,
                    source_timestamps=(sample.timestamp,),
                ),
                path=str(sample.path),
                source_timestamp=sample.timestamp,
            )
            candidates.append(
                FrameCandidate(
                    candidate_id=f"candidate_{index:03d}",
                    timestamp=sample.timestamp,
                    asset=asset,
                    perception=perception,
                )
            )

        return tuple(candidates)
