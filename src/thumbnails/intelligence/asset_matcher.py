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

            candidate_identifiers = set()
            for subject in perception.subjects.subjects:
                for attr in ("label", "name", "entity_id", "subject_id", "kind"):
                    val = getattr(subject, attr, None)
                    if isinstance(val, str) and val.strip():
                        candidate_identifiers.add(val.lower())

            evidence_score = 0.0
            total_evidence_categories = 0

            if concept.required_visual_evidence:
                total_evidence_categories += 1
                req_set = {item.lower() for item in concept.required_visual_evidence}
                evidence_score += len(candidate_identifiers.intersection(req_set)) / len(req_set)

            if concept.preferred_entities:
                total_evidence_categories += 1
                pref_ent_set = {item.lower() for item in concept.preferred_entities}
                evidence_score += len(candidate_identifiers.intersection(pref_ent_set)) / len(pref_ent_set)

            if concept.preferred_objects:
                total_evidence_categories += 1
                pref_obj_set = {item.lower() for item in concept.preferred_objects}
                evidence_score += len(candidate_identifiers.intersection(pref_obj_set)) / len(pref_obj_set)

            evidence_factor = (
                evidence_score / total_evidence_categories
                if total_evidence_categories
                else 0.0
            )

            if total_evidence_categories:
                score = 0.70 * base_score + 0.30 * evidence_factor
            else:
                score = base_score

            reasons = [
                f"quality={quality:.2f}",
                f"crop={crop:.2f}",
                f"focal={focal:.2f}",
            ]
            if total_evidence_categories:
                reasons.append(f"evidence_match={evidence_factor:.2f}")

            # When a concept describes a specific event, prefer frames from
            # that event instead of choosing a visually polished but unrelated
            # moment. The score decays smoothly outside the event window so a
            # nearby reaction frame can still compete.
            if concept.preferred_time_range is not None:
                start, end = concept.preferred_time_range
                timestamp = candidate.timestamp
                if start <= timestamp <= end:
                    temporal_score = 1.0
                else:
                    distance = start - timestamp if timestamp < start else timestamp - end
                    event_duration = max(end - start, 1.0)
                    temporal_score = max(0.0, 1.0 - distance / event_duration)
                score = 0.60 * score + 0.40 * temporal_score
                reasons.append(f"event_time_match={temporal_score:.2f}")

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
            sorted(matches, key=lambda match: match.suitability_score, reverse=True)
        )
