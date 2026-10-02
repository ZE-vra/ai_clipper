from __future__ import annotations

from pathlib import Path

from src.packaging.models import ClipPackaging
from src.thumbnails.domain.assets import AssetProvenance, VisualAsset
from src.thumbnails.domain.concepts import (
    CopyBlock,
    CopyConcept,
    CopyRole,
    ThumbnailConcept,
    VisualStrategy,
)
from src.thumbnails.domain.geometry import BoundingBox, Region, Size
from src.thumbnails.domain.results import ThumbnailResult, ThumbnailResultStatus
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.layout.composition import CompositionPlanner
from src.thumbnails.layout.negotiation import (
    LayoutCandidate,
    LayoutNegotiator,
)
from src.thumbnails.layout.typography import TypographyPlanner
from src.thumbnails.perception.frame_selector import SelectedFrame
from src.thumbnails.rendering.pillow_renderer import (
    PillowThumbnailRenderer,
    InMemoryVisualAssetResolver,
)


class ThumbnailPipeline:
    """Orchestrates the deterministic V1 thumbnail generation path."""

    def __init__(
        self,
        *,
        composition_planner: CompositionPlanner | None = None,
        typography_planner: TypographyPlanner | None = None,
        layout_negotiator: LayoutNegotiator | None = None,
    ) -> None:
        self._composition_planner = (
            composition_planner or CompositionPlanner()
        )
        self._typography_planner = (
            typography_planner or TypographyPlanner()
        )
        self._layout_negotiator = (
            layout_negotiator or LayoutNegotiator()
        )

    def generate(
        self,
        *,
        packaging: ClipPackaging,
        selected_frame: SelectedFrame,
        source_frame_path: str | Path,
        source_aspect_ratio: float,
        output_path: str | Path,
    ) -> ThumbnailResult:
        source_frame = Path(source_frame_path)

        if not source_frame.exists():
            raise FileNotFoundError(
                f"source frame does not exist: {source_frame}"
            )

        if source_aspect_ratio <= 0:
            raise ValueError(
                "source_aspect_ratio must be greater than 0."
            )

        if not packaging.thumbnail_text.strip():
            raise ValueError(
                "packaging.thumbnail_text must not be blank."
            )

        # ---------------------------------------------------------
        # 1. Creative bridge
        # ---------------------------------------------------------
        copy_block = CopyBlock(
            text=packaging.thumbnail_text,
            role=CopyRole.HOOK,
        )

        copy_concept = CopyConcept(
            concept_id=f"clip_{packaging.clip_id}_thumbnail_copy",
            blocks=(copy_block,),
            rationale=(
                "V1 bridge from clip packaging text into the "
                "thumbnail domain."
            ),
        )

        concept = ThumbnailConcept(
            concept_id=f"clip_{packaging.clip_id}_thumbnail",
            visual_strategy=VisualStrategy.SOURCE_FRAME,
            copy=copy_concept,
            rationale=(
                "V1 uses the selected source frame as the "
                "thumbnail's primary visual."
            ),
        )

        # Keep the concept creation explicit even though the
        # deterministic planners currently consume its components.
        del concept

        # ---------------------------------------------------------
        # 2. Asset boundary
        # ---------------------------------------------------------
        asset = VisualAsset(
            asset_id=f"clip_{packaging.clip_id}_thumbnail_source",
            provenance=AssetProvenance.SOURCE_FRAME,
            path=str(source_frame),
            source_timestamp=(
                selected_frame.score.timestamp
                if hasattr(selected_frame.score, "timestamp")
                else None
            ),
        )

        # ---------------------------------------------------------
        # 3. Target
        # ---------------------------------------------------------
        target = ThumbnailTarget(
            target_id=f"clip_{packaging.clip_id}_thumbnail_target",
            platform="vertical_video",
            size=Size(
                width=1080,
                height=1920,
            ),
            safe_regions=(
                Region(
                    name="primary_safe_region",
                    bounds=BoundingBox(
                        left=0.05,
                        top=0.05,
                        right=0.95,
                        bottom=0.95,
                    ),
                ),
            ),
        )

        # ---------------------------------------------------------
        # 4. Composition
        # ---------------------------------------------------------
        composition = self._composition_planner.plan(
            selected_frame=selected_frame,
            asset=asset,
            target=target,
            source_aspect_ratio=source_aspect_ratio,
        )

        # ---------------------------------------------------------
        # 5. Typography
        # ---------------------------------------------------------
        typography = self._typography_planner.plan(
            copy=copy_concept.blocks,
            composition=composition,
            target=target,
        )

        # ---------------------------------------------------------
        # 6. Render plan
        # ---------------------------------------------------------
        from src.thumbnails.domain.plans import (
            ThumbnailRenderPlan,
            VisualTreatmentPlan,
        )

        render_plan = ThumbnailRenderPlan(
            canvas_width=1080,
            canvas_height=1920,
            composition=composition,
            typography=typography,
            visual_treatment=VisualTreatmentPlan(
                contrast=1.05,
                saturation=1.05,
                sharpness=1.05,
                vignette=0.20,
                overlay_opacity=0.10,
            ),
        )

        # ---------------------------------------------------------
        # 7. Layout quality gate
        # ---------------------------------------------------------
        candidate = LayoutCandidate(
            candidate_id=f"clip_{packaging.clip_id}_layout_001",
            plan=render_plan,
            rationale=(
                "Initial deterministic V1 composition produced "
                "from the selected source frame."
            ),
        )

        negotiation = self._layout_negotiator.negotiate(
            candidates=(candidate,),
            target=target,
        )

        if negotiation.selected_candidate is None:
            return ThumbnailResult(
                status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
                failure_reason=(
                    negotiation.failure_reason
                    or "layout negotiation rejected the thumbnail."
                ),
            )

        selected_candidate = negotiation.selected_candidate

        # ---------------------------------------------------------
        # 8. Render
        # ---------------------------------------------------------
        resolver = InMemoryVisualAssetResolver(
            {
                asset.asset_id: asset,
            }
        )

        renderer = PillowThumbnailRenderer(
            asset_resolver=resolver,
        )

        output = renderer.render(
            plan=selected_candidate.plan,
            output_path=output_path,
        )

        return ThumbnailResult(
            status=ThumbnailResultStatus.SUCCESS,
            output_path=str(output),
            selected_attempt_id=selected_candidate.candidate_id,
        )


__all__ = ["ThumbnailPipeline"]

