from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.thumbnails.domain.concepts import VisualStrategy
from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentUnderstanding
from src.thumbnails.domain.plans import ThumbnailRenderPlan, VisualTreatmentPlan
from src.thumbnails.domain.results import ThumbnailResult, ThumbnailResultStatus
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.intelligence.asset_matcher import AssetMatcher
from src.thumbnails.intelligence.creative_director import CreativeDirector, RuleBasedCreativeDirector
from src.thumbnails.intelligence.strategy_planner import StrategyPlanner
from src.thumbnails.layout.composition import CompositionPlanner
from src.thumbnails.layout.negotiation import LayoutCandidate, LayoutNegotiator
from src.thumbnails.layout.typography import TypographyPlanner
from src.thumbnails.perception.frame_discovery import FrameDiscovery
from src.thumbnails.perception.frame_selector import SelectedFrame
from src.thumbnails.perception.frame_candidate_scorer import FrameCandidateScorer
from src.thumbnails.rendering.pillow_renderer import (
    InMemoryVisualAssetResolver,
    PillowThumbnailRenderer,
)


class ThumbnailRenderer(Protocol):
    def render(
        self,
        *,
        plan: ThumbnailRenderPlan,
        output_path: str | Path,
    ) -> Path:
        ...


class ThumbnailOrchestrator:
    """First real V2 vertical slice.

    Creative intent is produced before asset selection. Perception discovers
    the candidate pool, matching ranks assets per concept, and the existing
    deterministic composition/typography/renderer machinery executes the
    selected source-frame plan.

    This slice deliberately supports only SOURCE_FRAME execution. Other
    strategies
    remain represented in the domain so later builders can be added without
    changing the orchestration contract.
    """

    def __init__(
        self,
        *,
        frame_discovery: FrameDiscovery,
        creative_director: CreativeDirector | None = None,
        asset_matcher: AssetMatcher | None = None,
        strategy_planner: StrategyPlanner | None = None,
        composition_planner: CompositionPlanner | None = None,
        typography_planner: TypographyPlanner | None = None,
        layout_negotiator: LayoutNegotiator | None = None,
        renderer: ThumbnailRenderer | None = None,
        max_concepts: int = 3,
        max_matches_per_concept: int = 2,
    ) -> None:
        if max_concepts < 1:
            raise ValueError("max_concepts must be at least 1.")
        if max_matches_per_concept < 1:
            raise ValueError("max_matches_per_concept must be at least 1.")

        self.frame_discovery = frame_discovery
        self.creative_director = creative_director or RuleBasedCreativeDirector()
        self.asset_matcher = asset_matcher or AssetMatcher()
        self.strategy_planner = strategy_planner or StrategyPlanner()
        self.composition_planner = composition_planner or CompositionPlanner()
        self.typography_planner = typography_planner or TypographyPlanner()
        self.layout_negotiator = layout_negotiator or LayoutNegotiator()
        self.renderer = renderer
        self.max_concepts = max_concepts
        self.max_matches_per_concept = max_matches_per_concept

    def generate(
        self,
        *,
        source_video_path: str | Path,
        output_path: str | Path,
        candidates_dir: str | Path,
        brief: ThumbnailBrief,
        understanding: ContentUnderstanding,
        target: ThumbnailTarget,
    ) -> ThumbnailResult:
        source_video = Path(source_video_path)
        output = Path(output_path)

        if not source_video.is_file():
            return ThumbnailResult(
                status=ThumbnailResultStatus.FAILED,
                failure_reason=f"clean source video does not exist: {source_video}",
            )

        try:
            candidates = self.frame_discovery.discover(
                video_path=source_video,
                output_dir=Path(candidates_dir),
            )
            if not candidates:
                return ThumbnailResult(
                    status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
                    failure_reason="frame discovery produced no candidates",
                )

            concepts = self.creative_director.create(
                brief=brief,
                understanding=understanding,
                max_concepts=self.max_concepts,
            )
            if not concepts:
                return ThumbnailResult(
                    status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
                    failure_reason="creative director produced no concepts",
                )

            assets = {candidate.asset.asset_id: candidate.asset for candidate in candidates}

            layout_candidates: list[LayoutCandidate] = []

            for concept in concepts:
                if concept.copy is None:
                    continue

                matches = self.asset_matcher.match(
                    concept=concept,
                    candidates=candidates,
                )[: self.max_matches_per_concept]

                for match in matches:
                    candidate = next(
                        candidate
                        for candidate in candidates
                        if candidate.asset.asset_id == match.asset_id
                    )
                    if candidate.perception is None:
                        continue

                    asset_plan = self.strategy_planner.plan(
                        concept=concept,
                        match=match,
                        candidates=candidates,
                    )

                    # The first executable slice is intentionally source-frame
                    # only. Unsupported strategies remain candidates for later
                    # builders rather than being silently faked here.
                    if asset_plan.strategy is not VisualStrategy.SOURCE_FRAME:
                        continue

                    try:
                        selected = SelectedFrame(
                            perception=candidate.perception,
                            score=FrameCandidateScorer().score(candidate.perception),
                        )

                        source_aspect_ratio = self._aspect_ratio(
                            candidate.asset.path
                        )

                        composition = self.composition_planner.plan(
                            selected_frame=selected,
                            asset=candidate.asset,
                            target=target,
                            source_aspect_ratio=source_aspect_ratio,
                        )

                        typography = self.typography_planner.plan(
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
                                contrast=1.06,
                                saturation=1.06,
                                sharpness=1.04,
                                vignette=0.08,
                            ),
                        )
                    except (TypeError, ValueError, FileNotFoundError):
                        # One bad candidate must not destroy the remaining search
                        # space. The evaluator/orchestrator can only choose from
                        # plans that successfully materialize.
                        continue

                    layout_candidates.append(
                        LayoutCandidate(
                            candidate_id=asset_plan.plan_id,
                            plan=render_plan,
                            rationale=asset_plan.rationale,
                        )
                    )

            negotiation = self.layout_negotiator.negotiate(
                candidates=layout_candidates,
                target=target,
            )

            if negotiation.selected_candidate is None:
                return ThumbnailResult(
                    status=ThumbnailResultStatus.NO_ACCEPTABLE_RESULT,
                    failure_reason=negotiation.failure_reason,
                )

            selected_plan = negotiation.selected_candidate.plan
            resolver = InMemoryVisualAssetResolver(assets)
            renderer = self.renderer or PillowThumbnailRenderer(
                asset_resolver=resolver,
            )

            renderer.render(
                plan=selected_plan,
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
    def _aspect_ratio(path: str | Path) -> float:
        from PIL import Image

        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                raise ValueError("source frame has invalid dimensions.")
            return image.width / image.height
