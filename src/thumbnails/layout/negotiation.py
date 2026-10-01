from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from src.thumbnails.domain.geometry import BoundingBox, Point
from src.thumbnails.domain.concepts import CopyRole
from src.thumbnails.domain.plans import ThumbnailRenderPlan
from src.thumbnails.domain.target import ThumbnailTarget


@dataclass(frozen=True)
class LayoutCandidate:
    """A complete deterministic layout that can compete for selection."""

    candidate_id: str
    plan: ThumbnailRenderPlan
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank.")


@dataclass(frozen=True)
class LayoutEvaluation:
    """Deterministic evaluation of one layout candidate."""

    candidate_id: str
    accepted: bool
    score: float
    hard_failures: tuple[str, ...] = ()
    soft_failures: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank.")
        if not 0.0 <= self.score <= 100.0:
            raise ValueError("score must be between 0 and 100.")


@dataclass(frozen=True)
class LayoutNegotiationResult:
    """Outcome of comparing a bounded set of layout candidates."""

    selected_candidate: Optional[LayoutCandidate]
    evaluations: tuple[LayoutEvaluation, ...]
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.selected_candidate is None and not self.failure_reason:
            raise ValueError(
                "failure_reason is required when no candidate is selected."
            )
        if self.selected_candidate is not None and self.failure_reason:
            raise ValueError(
                "failure_reason must be empty when a candidate is selected."
            )


@dataclass(frozen=True)
class LayoutNegotiatorConfig:
    """Policy controlling deterministic layout selection."""

    minimum_acceptable_score: float = 70.0
    max_candidates: int = 8
    ui_overlap_tolerance: float = 0.05
    focal_point_exclusion_radius: float = 0.12
    minimum_primary_font_size: int = 54
    maximum_word_count: int = 5
    hierarchy_tolerance: float = 1.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_acceptable_score <= 100.0:
            raise ValueError("minimum_acceptable_score must be between 0 and 100.")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than 0.")
        if not 0.0 <= self.ui_overlap_tolerance <= 1.0:
            raise ValueError("ui_overlap_tolerance must be between 0 and 1.")
        if self.focal_point_exclusion_radius < 0.0:
            raise ValueError("focal_point_exclusion_radius must not be negative.")
        if self.minimum_primary_font_size <= 0:
            raise ValueError("minimum_primary_font_size must be greater than 0.")
        if self.maximum_word_count <= 0:
            raise ValueError("maximum_word_count must be greater than 0.")
        if self.hierarchy_tolerance < 1.0:
            raise ValueError("hierarchy_tolerance must be at least 1.")


class LayoutEvaluator:
    """Scores layout quality using geometry and typography evidence only."""

    def __init__(self, config: LayoutNegotiatorConfig | None = None) -> None:
        self.config = config or LayoutNegotiatorConfig()

    def evaluate(
        self,
        *,
        candidate: LayoutCandidate,
        target: ThumbnailTarget,
    ) -> LayoutEvaluation:
        plan = candidate.plan
        hard_failures: list[str] = []
        soft_failures: list[str] = []

        # Start from a neutral structural score rather than 100. Scores are
        # assembled from independent quality components below so stronger
        # candidates can actually distinguish themselves.
        score = 50.0

        if plan.canvas_width != target.size.width or plan.canvas_height != target.size.height:
            hard_failures.append("canvas dimensions do not match target")

        placements = plan.composition.visual_placements
        primary_point = placements[0].focal_point if placements else Point(0.5, 0.5)

        ui_overlap = 0.0
        safe_coverage = 1.0
        focal_clearance = 1.0

        blocks = plan.typography.blocks
        for block in blocks:
            bounds = block.text_bounds

            if not self._inside_canvas(bounds):
                hard_failures.append(f"{block.copy.text!r} extends outside the canvas")

            if target.safe_regions and not any(
                self._contains(region.bounds, bounds) for region in target.safe_regions
            ):
                hard_failures.append(
                    f"{block.copy.text!r} is outside every target safe region"
                )

            block_ui_overlap = max(
                (
                    self._weighted_overlap(bounds, occlusion.region.bounds, occlusion.weight)
                    for occlusion in target.ui_occlusion_regions
                ),
                default=0.0,
            )
            ui_overlap = max(ui_overlap, block_ui_overlap)
            if block_ui_overlap > self.config.ui_overlap_tolerance:
                hard_failures.append(
                    f"{block.copy.text!r} overlaps protected UI"
                )

            if self._contains(bounds, primary_point):
                hard_failures.append(
                    f"{block.copy.text!r} covers the primary focal point"
                )

            clearance = self._point_to_box_distance(primary_point, bounds)
            if clearance < focal_clearance:
                focal_clearance = clearance

            if clearance < self.config.focal_point_exclusion_radius:
                soft_failures.append(
                    f"{block.copy.text!r} is close to the primary focal point"
                )

        typography_score = 0.0
        copy_score = 0.0
        hierarchy_bonus = 0.0

        if blocks:
            primary_font = max(block.font_size for block in blocks)

            # Typography size contributes up to 20 points. Reaching the
            # configured minimum earns half credit; larger type earns more
            # until the component is saturated.
            font_ratio = min(
                1.0,
                primary_font
                / (self.config.minimum_primary_font_size * 2.0),
            )
            typography_score = font_ratio * 20.0

            if primary_font < self.config.minimum_primary_font_size:
                soft_failures.append("primary typography is too small")

            total_words = sum(
                len((block.rendered_text or block.copy.text).split())
                for block in blocks
            )

            if total_words <= self.config.maximum_word_count:
                copy_score = 15.0
            else:
                soft_failures.append("thumbnail copy is too long")
                excess_words = total_words - self.config.maximum_word_count
                copy_score = max(0.0, 15.0 - excess_words * 5.0)

            hierarchy_score = self._hierarchy_score(blocks)
            if hierarchy_score < 1.0:
                soft_failures.append("typography hierarchy is too flat")
            else:
                hierarchy_bonus = min(
                    15.0,
                    max(0.0, (hierarchy_score - 1.0) * 15.0),
                )

            score += typography_score + copy_score + hierarchy_bonus
        else:
            hard_failures.append("typography contains no blocks")

        if ui_overlap > 0.0:
            score -= min(20.0, ui_overlap * 100.0)

        if focal_clearance < self.config.focal_point_exclusion_radius:
            score -= 8.0

        if hard_failures:
            score = 0.0

        accepted = not hard_failures and score >= self.config.minimum_acceptable_score

        if not accepted and not hard_failures:
            soft_failures.append("layout score is below acceptance threshold")

        return LayoutEvaluation(
            candidate_id=candidate.candidate_id,
            accepted=accepted,
            score=max(0.0, min(100.0, score)),
            hard_failures=tuple(dict.fromkeys(hard_failures)),
            soft_failures=tuple(dict.fromkeys(soft_failures)),
            metrics={
                "ui_overlap": ui_overlap,
                "focal_clearance": focal_clearance,
                "safe_coverage": safe_coverage,
                "typography_score": typography_score,
                "copy_score": copy_score,
                "hierarchy_bonus": hierarchy_bonus,
            },
        )

    def _hierarchy_score(self, blocks) -> float:
        if len(blocks) < 2:
            return 1.5

        primary_roles = {CopyRole.PAYOFF, CopyRole.HOOK}
        primary_sizes = [
            block.font_size for block in blocks if block.copy.role in primary_roles
        ]
        if not primary_sizes:
            primary_sizes = [block.font_size for block in blocks]

        largest = max(block.font_size for block in blocks)
        second = sorted((block.font_size for block in blocks), reverse=True)[1]
        if largest <= 0 or second <= 0:
            return 0.0
        return largest / second

    @staticmethod
    def _inside_canvas(bounds: BoundingBox) -> bool:
        return (
            0.0 <= bounds.left <= bounds.right <= 1.0
            and 0.0 <= bounds.top <= bounds.bottom <= 1.0
        )

    @staticmethod
    def _contains(container: BoundingBox, item: BoundingBox | Point) -> bool:
        if isinstance(item, Point):
            return (
                container.left <= item.x <= container.right
                and container.top <= item.y <= container.bottom
            )
        return (
            container.left <= item.left
            and item.right <= container.right
            and container.top <= item.top
            and item.bottom <= container.bottom
        )

    @staticmethod
    def _weighted_overlap(
        a: BoundingBox,
        b: BoundingBox,
        weight: float,
    ) -> float:
        width = max(0.0, min(a.right, b.right) - max(a.left, b.left))
        height = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
        overlap = width * height
        return min(1.0, overlap / (a.width * a.height) * weight)

    @staticmethod
    def _point_to_box_distance(point: Point, box: BoundingBox) -> float:
        dx = max(box.left - point.x, 0.0, point.x - box.right)
        dy = max(box.top - point.y, 0.0, point.y - box.bottom)
        return (dx * dx + dy * dy) ** 0.5


class LayoutNegotiator:
    """
    Selects the strongest acceptable layout from a bounded candidate set.

    Candidate generation remains outside this class. That keeps creative
    generation and deterministic decision-making separate.
    """

    def __init__(
        self,
        evaluator: LayoutEvaluator | None = None,
        config: LayoutNegotiatorConfig | None = None,
    ) -> None:
        self.config = config or LayoutNegotiatorConfig()
        self.evaluator = evaluator or LayoutEvaluator(self.config)

    def negotiate(
        self,
        *,
        candidates: Iterable[LayoutCandidate],
        target: ThumbnailTarget,
    ) -> LayoutNegotiationResult:
        materialized = tuple(candidates)
        if not materialized:
            return LayoutNegotiationResult(
                selected_candidate=None,
                evaluations=(),
                failure_reason="no layout candidates were provided",
            )

        if len(materialized) > self.config.max_candidates:
            materialized = materialized[: self.config.max_candidates]

        ids = [candidate.candidate_id for candidate in materialized]
        if len(ids) != len(set(ids)):
            raise ValueError("layout candidate IDs must be unique.")

        evaluations = tuple(
            self.evaluator.evaluate(candidate=candidate, target=target)
            for candidate in materialized
        )

        accepted = [
            (candidate, evaluation, index)
            for index, (candidate, evaluation) in enumerate(
                zip(materialized, evaluations)
            )
            if evaluation.accepted
        ]

        if not accepted:
            best = max(evaluations, key=lambda evaluation: evaluation.score)
            return LayoutNegotiationResult(
                selected_candidate=None,
                evaluations=evaluations,
                failure_reason=(
                    f"no acceptable layout; best candidate "
                    f"{best.candidate_id!r} scored {best.score:.1f}"
                ),
            )

        selected, _, _ = max(
            accepted,
            key=lambda item: (item[1].score, -item[2]),
        )
        return LayoutNegotiationResult(
            selected_candidate=selected,
            evaluations=evaluations,
        )
