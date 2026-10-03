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

            base_score = (
                0.35 * quality
                + 0.30 * crop
                + 0.20 * focal
                + 0.15 * min(1.0, subjects / 3.0)
            )

            # Extract candidate identifiers from subjects safely
            candidate_identifiers = set()
            if hasattr(perception, "subjects") and hasattr(perception.subjects, "subjects"):
                for subject in perception.subjects.subjects:
                    for attr in ("label", "name", "entity_id", "subject_id", "kind"):
                        val = getattr(subject, attr, None)
                        if isinstance(val, str) and val.strip():
                            candidate_identifiers.add(val.lower())

            # Calculate evidence matching score
            evidence_score = 0.0
            total_evidence_categories = 0

            if concept.required_visual_evidence:
                total_evidence_categories += 1
                req_set = {item.lower() for item in concept.required_visual_evidence}
                req_matches = candidate_identifiers.intersection(req_set)
                evidence_score += len(req_matches) / len(req_set)

            if concept.preferred_entities:
                total_evidence_categories += 1
                pref_ent_set = {item.lower() for item in concept.preferred_entities}
                pref_ent_matches = candidate_identifiers.intersection(pref_ent_set)
                evidence_score += len(pref_ent_matches) / len(pref_ent_set)

            if concept.preferred_objects:
                total_evidence_categories += 1
                pref_obj_set = {item.lower() for item in concept.preferred_objects}
                pref_obj_matches = candidate_identifiers.intersection(pref_obj_set)
                evidence_score += len(pref_obj_matches) / len(pref_obj_set)

            if total_evidence_categories > 0:
                evidence_factor = evidence_score / total_evidence_categories
                score = 0.70 * base_score + 0.30 * evidence_factor
            else:
                evidence_factor = 0.0
                score = base_score

            reasons = [
                f"quality={quality:.2f}",
                f"crop={crop:.2f}",
                f"focal={focal:.2f}",
            ]
            if total_evidence_categories > 0:
                reasons.append(f"evidence_match={evidence_factor:.2f}")

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
