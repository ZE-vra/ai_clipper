from __future__ import annotations

from src.thumbnails.domain.assets import AssetMatch, FrameCandidate
from src.thumbnails.domain.concepts import ThumbnailConcept, VisualStrategy
from src.thumbnails.domain.operations import AssetOperation, AssetOperationKind
from src.thumbnails.domain.plans import AssetPlan


class StrategyPlanner:
    """Turns a concept/asset match into an executable asset recipe."""

    def plan(
        self,
        *,
        concept: ThumbnailConcept,
        match: AssetMatch,
        candidates: tuple[FrameCandidate, ...],
    ) -> AssetPlan:
        candidate = next(
            candidate
            for candidate in candidates
            if candidate.asset.asset_id == match.asset_id
        )

        strategy = self._select_strategy(concept)
        operations = [
            AssetOperation(
                operation_id=f"{match.asset_id}-extract",
                kind=AssetOperationKind.EXTRACT,
                input_asset_ids=(candidate.asset.asset_id,),
                rationale="Use the discovered source frame as the initial visual asset.",
            )
        ]

        if strategy is VisualStrategy.ENHANCED_FRAME:
            operations.append(
                AssetOperation(
                    operation_id=f"{match.asset_id}-enhance",
                    kind=AssetOperationKind.ENHANCE,
                    input_asset_ids=(candidate.asset.asset_id,),
                    rationale="Improve an otherwise viable source frame deterministically.",
                )
            )
        elif strategy is VisualStrategy.SUBJECT_CUTOUT:
            operations.append(
                AssetOperation(
                    operation_id=f"{match.asset_id}-segment",
                    kind=AssetOperationKind.SEGMENT,
                    input_asset_ids=(candidate.asset.asset_id,),
                    rationale="Separate the subject for independent composition.",
                )
            )

        return AssetPlan(
            plan_id=f"{concept.concept_id}-{match.asset_id}-{strategy.value}",
            concept_id=concept.concept_id,
            strategy=strategy,
            source_asset_ids=(candidate.asset.asset_id,),
            operations=tuple(operations),
            requirements=concept.required_visual_evidence,
            rationale=(
                f"Matched asset {match.asset_id} to concept {concept.concept_id} "
                f"with suitability {match.suitability_score:.2f}."
            ),
            estimated_cost=0.0,
            plan_fingerprint=(
                f"{concept.concept_id}|{match.asset_id}|{strategy.value}"
            ),
        )

    @staticmethod
    def _select_strategy(concept: ThumbnailConcept) -> VisualStrategy:
        executable_strategies = (
            VisualStrategy.SOURCE_FRAME,
            VisualStrategy.ENHANCED_FRAME,
            VisualStrategy.SUBJECT_CUTOUT,
        )

        # Legacy concepts may specify one explicit strategy without a candidate list.
        if concept.visual_strategy is not None:
            if concept.visual_strategy not in executable_strategies:
                raise ValueError(
                    f"Strategy {concept.visual_strategy.value!r} is not executable "
                    "by StrategyPlanner."
                )
            return concept.visual_strategy

        for strategy in executable_strategies:
            if strategy in concept.candidate_strategies:
                return strategy

        raise ValueError(
            f"Concept {concept.concept_id!r} has no executable candidate strategy."
        )
