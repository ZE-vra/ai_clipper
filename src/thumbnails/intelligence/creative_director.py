from __future__ import annotations

from typing import Protocol

from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.concepts import (
    CopyBlock,
    CopyConcept,
    CopyRole,
    ThumbnailConcept,
    VisualStrategy,
)
from src.thumbnails.domain.content import ContentUnderstanding


class CreativeDirector(Protocol):
    """Produces creative concepts without selecting a source frame."""

    def create(
        self,
        *,
        brief: ThumbnailBrief,
        understanding: ContentUnderstanding,
        max_concepts: int,
    ) -> tuple[ThumbnailConcept, ...]:
        ...


class RuleBasedCreativeDirector:
    """Deterministic first vertical-slice director.

    This intentionally makes no frame decision. Its job is to prove the
    architecture: creative intent exists before asset selection.
    """

    def create(
        self,
        *,
        brief: ThumbnailBrief,
        understanding: ContentUnderstanding,
        max_concepts: int,
    ) -> tuple[ThumbnailConcept, ...]:
        if max_concepts < 1:
            return ()

        strategies = (
            VisualStrategy.SOURCE_FRAME,
            VisualStrategy.ENHANCED_FRAME,
            VisualStrategy.SUBJECT_CUTOUT,
            VisualStrategy.HYBRID,
            VisualStrategy.GENERATED_VISUAL,
        )

        concepts: list[ThumbnailConcept] = []

        if brief.important_objects:
            object_name = brief.important_objects[0]
            concepts.append(
                ThumbnailConcept(
                    concept_id="object-reveal",
                    title=f"The {object_name} is the story",
                    visual_idea=(
                        f"Make the most visually distinctive view of {object_name} "
                        "the dominant element."
                    ),
                    copy=CopyConcept(
                        concept_id="object-reveal-copy",
                        blocks=(
                            CopyBlock(
                                brief.core_hook,
                                CopyRole.HOOK,
                            ),
                        ),
                    ),
                    curiosity_mechanism=brief.curiosity_angle,
                    emotional_direction=brief.emotional_direction or "surprise",
                    required_visual_evidence=(object_name,),
                    preferred_objects=(object_name,),
                    candidate_strategies=strategies,
                    composition_direction="Dominant object with clean supporting space.",
                    rationale="Object-led concept derived from the creative brief.",
                    priority=1,
                )
            )

        concepts.append(
            ThumbnailConcept(
                concept_id="core-hook",
                title="Core hook",
                visual_idea=(
                    "Use the strongest truthful visual evidence for the central hook "
                    "and make the unanswered question visually obvious."
                ),
                copy=CopyConcept(
                    concept_id="core-hook-copy",
                    blocks=(
                        CopyBlock(
                            brief.core_hook,
                            CopyRole.HOOK,
                        ),
                    ),
                ),
                curiosity_mechanism=brief.curiosity_angle,
                emotional_direction=brief.emotional_direction or "curiosity",
                required_visual_evidence=brief.visual_evidence,
                preferred_entities=brief.important_entities,
                candidate_strategies=strategies,
                composition_direction="Strong focal subject with deliberate text separation.",
                rationale="Hook-led concept that preserves multiple asset strategies.",
                priority=2,
            )
        )

        if understanding.events:
            event = max(
                understanding.events,
                key=lambda candidate: candidate.importance * candidate.confidence,
            )

            entity_map = {entity.entity_id: entity.label for entity in understanding.entities}
            resolved_labels = [entity_map[eid] for eid in event.entity_ids if eid in entity_map]

            visual_idea = (
                f"Represent the event '{event.description}' with a frame that makes "
                "the viewer want to know what happened immediately before or after it."
            )
            if resolved_labels:
                visual_idea += f" Focus on key entities: {', '.join(resolved_labels)}."

            concepts.append(
                ThumbnailConcept(
                    concept_id="event-reveal",
                    title=f"The revealing moment: {event.description}",
                    visual_idea=visual_idea,
                    copy=CopyConcept(
                        concept_id="event-reveal-copy",
                        blocks=(
                            CopyBlock(
                                brief.promise,
                                CopyRole.PAYOFF,
                            ),
                        ),
                    ),
                    curiosity_mechanism=brief.curiosity_angle,
                    emotional_direction=brief.emotional_direction or "reveal",
                    required_visual_evidence=event.entity_ids,
                    preferred_entities=event.entity_ids,
                    candidate_strategies=strategies,
                    composition_direction="Event-led composition with clear visual hierarchy.",
                    rationale="Temporal event selected before any frame is selected.",
                    priority=3,
                )
            )

        return tuple(concepts[:max_concepts])
