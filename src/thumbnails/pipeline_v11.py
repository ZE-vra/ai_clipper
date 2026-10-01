from __future__ import annotations

from pathlib import Path

from src.packaging.models import ClipPackaging

from src.thumbnails.domain.assets import AssetProvenance, VisualAsset
from src.thumbnails.domain.concepts import ThumbnailConcept, VisualStrategy
from src.thumbnails.domain.geometry import BoundingBox, Region, Size
from src.thumbnails.domain.plans import ThumbnailRenderPlan, VisualTreatmentPlan
from src.thumbnails.domain.results import ThumbnailResult, ThumbnailResultStatus
from src.thumbnails.domain.target import ThumbnailTarget, UIOcclusionRegion
from src.thumbnails.frame_extractor import FFmpegFrameExtractor
from src.thumbnails.layout.negotiation import (
    LayoutCandidate,
    LayoutNegotiator,
    LayoutNegotiatorConfig,
)
from src.thumbnails.layout.typography import TypographyPlanner, TypographyPlannerConfig
from src.thumbnails.layout.v11_composition import V11CompositionPlanner
from src.thumbnails.layout.v11_copy import V11CopyDirector
from src.thumbnails.perception.crop_analyzer import CropAnalyzer
from src.thumbnails.perception.focal_analyzer import FocalRegionAnalyzer
from src.thumbnails.perception.frame_perception_analyzer import FramePerceptionAnalyzer
from src.thumbnails.perception.frame_selector import FrameSelector
from src.thumbnails.perception.local_subject_analyzer import LocalSubjectAnalyzer
from src.thumbnails.perception.quality_analyzer import FrameQualityAnalyzer
from src.thumbnails.perception.video_frame_sampler import (
    FFprobeVideoDurationReader,
    ThumbnailFrameSampler,
)
from src.thumbnails.rendering.v11_renderer import (
    V11InMemoryAssetResolver,
    V11PillowThumbnailRenderer,
)


class ThumbnailPipelineV11:
    """
    End-to-end V1.1 thumbnail generation from a CLEAN source video section.

    The key boundary is deliberate: this pipeline never accepts the already
    edited vertical clip as its visual source. It samples the clean source
    section before captions and vertical composition are applied.
    """

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        yolo_model_path: str | Path = "yolo26n.pt",
    ) -> None:
        self.sampler = ThumbnailFrameSampler(
            extractor=FFmpegFrameExtractor(
                ffmpeg_binary=ffmpeg_binary,
            ),
            duration_reader=FFprobeVideoDurationReader(
                ffprobe_binary=ffprobe_binary,
            ).read,
            sample_count=9,
        )

        self.perception = FramePerceptionAnalyzer(
            quality_analyzer=FrameQualityAnalyzer(),
            focal_analyzer=FocalRegionAnalyzer(),
            subject_analyzer=LocalSubjectAnalyzer(yolo_model_path),
            crop_analyzer=CropAnalyzer(),
        )

        self.selector = FrameSelector()
        self.copy_director = V11CopyDirector()
        self.composition = V11CompositionPlanner()

        self.typography = TypographyPlanner(
            TypographyPlannerConfig(
                base_font_size=120,
                minimum_font_size=54,
                maximum_font_size=148,
                hook_scale=0.92,
                payoff_scale=1.0,
                context_scale=0.60,
                accent_scale=0.50,
                max_lines_per_block=2,
                line_spacing=0.88,
                block_gap=0.018,
                horizontal_padding=0.035,
                vertical_padding=0.02,
                minimum_region_width=0.35,
                minimum_region_height=0.12,
            )
        )

        self.negotiator = LayoutNegotiator(
            config=LayoutNegotiatorConfig(
                minimum_acceptable_score=70.0,
                minimum_primary_font_size=54,
                maximum_word_count=5,
            )
        )

    def generate(
        self,
        *,
        packaging: ClipPackaging,
        source_video_path: str | Path,
        output_path: str | Path,
    ) -> ThumbnailResult:
        source_video = Path(source_video_path)
        output = Path(output_path)

        if not source_video.is_file():
            return ThumbnailResult(
                status=ThumbnailResultStatus.FAILED,
                failure_reason=f"clean source video does not exist: {source_video}",
            )

        try:
            candidates_dir = output.parent / f"{output.stem}_candidates"
            samples = self.sampler.sample(
                video_path=source_video,
                output_dir=candidates_dir,
            )

            perceptions = tuple(
                self.perception.analyze(sample.path)
                for sample in samples
            )

            selected = self.selector.select(list(perceptions))

            selected_sample = next(
                sample
                for sample in samples
                if sample.path == selected.perception.frame_path
            )

            copy = self.copy_director.create(
                text=packaging.thumbnail_text,
                concept_id=f"clip_{packaging.clip_id}_v11_copy",
            )

            concept = ThumbnailConcept(
                concept_id=f"clip_{packaging.clip_id}_v11",
                visual_strategy=VisualStrategy.SOURCE_FRAME,
                copy=copy,
                rationale=(
                    "Clean-source, subject-first vertical composition "
                    "with deterministic typography hierarchy."
                ),
                priority=1,
            )

            asset = VisualAsset(
                asset_id=f"clip_{packaging.clip_id}_frame_{selected_sample.timestamp:.3f}",
                provenance=AssetProvenance.SOURCE_FRAME,
                path=str(selected_sample.path),
                source_timestamp=selected_sample.timestamp,
            )

            target = self._target()
            source_aspect_ratio = self._aspect_ratio(selected_sample.path)

            composition = self.composition.plan(
                selected_frame=selected,
                asset=asset,
                target=target,
                source_aspect_ratio=source_aspect_ratio,
            )

            typography = self.typography.plan(
                copy=concept.copy.blocks,
                composition=composition,
                target=target,
            )

            render_plan = ThumbnailRenderPlan(
                canvas_width=target.size.width,
                canvas_height=target.size.height,
                composition=composition,
                typography=typography,
                visual_treatment=VisualTreatmentPlan(
                    contrast=1.10,
                    saturation=1.10,
                    sharpness=1.08,
                    vignette=0.12,
                    overlay_opacity=0.0,
                ),
            )

            candidate = LayoutCandidate(
                candidate_id=f"clip_{packaging.clip_id}_v11_layout_001",
                plan=render_plan,
                rationale=(
                    "Subject-first 9:16 crop with dedicated upper text band."
                ),
            )

            negotiation = self.negotiator.negotiate(
                candidates=(candidate,),
                target=target,
            )

            if negotiation.selected_candidate is None:
                return ThumbnailResult(
                    status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
                    failure_reason=negotiation.failure_reason,
                )

            renderer = V11PillowThumbnailRenderer(
                asset_resolver=V11InMemoryAssetResolver(
                    {asset.asset_id: asset}
                )
            )

            renderer.render(
                plan=negotiation.selected_candidate.plan,
                output_path=output,
            )

            return ThumbnailResult(
                status=ThumbnailResultStatus.SUCCESS,
                output_path=str(output),
                selected_attempt_id=negotiation.selected_candidate.candidate_id,
            )

        except Exception as exc:
            return ThumbnailResult(
                status=ThumbnailResultStatus.FAILED,
                failure_reason=str(exc),
            )

    @staticmethod
    def _aspect_ratio(path: Path) -> float:
        from PIL import Image

        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                raise ValueError("source frame has invalid dimensions.")
            return image.width / image.height

    @staticmethod
    def _target() -> ThumbnailTarget:
        return ThumbnailTarget(
            target_id="youtube-shorts-v11",
            platform="youtube_shorts",
            size=Size(width=1080, height=1920),
            safe_regions=(
                Region(
                    name="primary_safe_area",
                    bounds=BoundingBox(
                        left=0.04,
                        top=0.04,
                        right=0.96,
                        bottom=0.82,
                    ),
                ),
            ),
            ui_occlusion_regions=(
                UIOcclusionRegion(
                    region=Region(
                        name="right_action_stack",
                        bounds=BoundingBox(
                            left=0.86,
                            top=0.12,
                            right=1.0,
                            bottom=0.78,
                        ),
                    ),
                    weight=1.0,
                    reason="Shorts action controls may overlay this edge.",
                ),
                UIOcclusionRegion(
                    region=Region(
                        name="bottom_metadata",
                        bounds=BoundingBox(
                            left=0.0,
                            top=0.82,
                            right=1.0,
                            bottom=1.0,
                        ),
                    ),
                    weight=1.0,
                    reason="Shorts metadata and controls may overlay the bottom.",
                ),
            ),
        )
