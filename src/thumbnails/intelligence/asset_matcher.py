from __future__ import annotations

from src.thumbnails.domain.assets import AssetMatch, FrameCandidate
from src.thumbnails.domain.concepts import ThumbnailConcept, VisualStrategy


class AssetMatcher:
    """Matches discovered assets to concepts without making a global choice."""

    def match(
        self,
        *,
        concept: ThumbnailConcept,
        candidates: tuple[FrameCandidate, ...],
    ) -> tuple[AssetMatch, ...]:
        matches: list[AssetMatch] = []

        for candidate in candidates:
            perception = candidate.perception
            if perception is None:
                matches.append(
                    AssetMatch(
                        asset_id=candidate.asset.asset_id,
                        concept_id=concept.concept_id,
                        suitability_score=0.0,
                        limitations=("candidate has no perception evidence",),
                    )
                )
                continue

            quality = perception.quality.overall_quality
            crop = perception.crop.score
            focal = max(
                (region.strength for region in perception.focal.regions),
                default=0.0,
            )
            subjects = len(perception.subjects.subjects)

            score = (
                0.35 * quality
                + 0.30 * crop
                + 0.20 * focal
                + 0.15 * min(1.0, subjects / 3.0)
            )

            reasons = [
                f"quality={quality:.2f}",
                f"crop={crop:.2f}",
                f"focal={focal:.2f}",
            ]
            limitations: list[str] = []

            if not perception.subjects.subjects:
                limitations.append("no detected person subject")

            if VisualStrategy.SOURCE_FRAME not in concept.candidate_strategies:
                limitations.append("source frame is not a preferred strategy")

            matches.append(
                AssetMatch(
                    asset_id=candidate.asset.asset_id,
                    concept_id=concept.concept_id,
                    suitability_score=max(0.0, min(1.0, score)),
                    reasons=tuple(reasons),
                    limitations=tuple(limitations),
                )
            )

        return tuple(
            sorted(
                matches,
                key=lambda match: match.suitability_score,
                reverse=True,
            )
        )
