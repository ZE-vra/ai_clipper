from src.thumbnails.domain.assets import (
    AssetProvenance,
    AssetProvenanceKind,
)
from src.thumbnails.domain.content import (
    ContentEntity,
    ContentEvent,
    ContentUnderstanding,
)
from src.thumbnails.domain.concepts import (
    CopyBlock,
    CopyConcept,
    CopyRole,
    ThumbnailConcept,
    VisualStrategy,
)
from src.thumbnails.domain.diagnosis import (
    EvaluationDiagnosis,
    EvaluationFailureKind,
)
from src.thumbnails.domain.evaluation import (
    IdentityConstraint,
    PromiseAlignment,
    RepresentationMode,
    ThumbnailEvaluation,
)
from src.thumbnails.domain.operations import (
    AssetOperation,
    AssetOperationKind,
)
from src.thumbnails.domain.search import SearchBudget, SearchState


def test_content_understanding_preserves_entities_and_events() -> None:
    entity = ContentEntity("plane", "private jet", "object", 0.98)
    event = ContentEvent(
        "reveal",
        12.0,
        18.0,
        "The jet interior is revealed.",
        entity_ids=("plane",),
        importance=0.9,
        confidence=0.95,
    )

    understanding = ContentUnderstanding(
        entities=(entity,),
        events=(event,),
        themes=("luxury",),
        claims=("The jet has a $2M carpet.",),
        confidence=0.9,
    )

    assert understanding.events[0].entity_ids == ("plane",)


def test_concept_can_describe_strategy_options_without_selecting_a_frame() -> None:
    copy = CopyConcept(
        "copy-1",
        (CopyBlock("$2M CARPET", CopyRole.HOOK),),
    )
    concept = ThumbnailConcept(
        concept_id="concept-1",
        title="The impossible carpet",
        visual_idea="Make the absurd carpet value the visual focal point.",
        copy=copy,
        curiosity_mechanism="The price feels unbelievable.",
        emotional_direction="surprise",
        preferred_objects=("carpet", "jet interior"),
        candidate_strategies=(
            VisualStrategy.SOURCE_FRAME,
            VisualStrategy.ENHANCED_FRAME,
            VisualStrategy.GENERATED_VISUAL,
        ),
    )

    assert concept.visual_strategy is None
    assert VisualStrategy.GENERATED_VISUAL in concept.candidate_strategies


def test_asset_provenance_records_lineage() -> None:
    provenance = AssetProvenance(
        kind=AssetProvenanceKind.GENERATED,
        source_asset_ids=("frame-12",),
        source_timestamps=(12.4,),
        generator="test-provider",
        generation_prompt_id="prompt-v1",
    )

    assert provenance.kind is AssetProvenanceKind.GENERATED
    assert provenance.source_asset_ids == ("frame-12",)


def test_asset_operations_are_explicit() -> None:
    operation = AssetOperation(
        operation_id="op-1",
        kind=AssetOperationKind.COMPOSITE,
        input_asset_ids=("frame-1", "generated-1"),
        parameters=(("mode", "overlay"),),
    )

    assert operation.kind is AssetOperationKind.COMPOSITE


def test_truth_contract_is_explicit() -> None:
    constraint = IdentityConstraint("person-1")
    alignment = PromiseAlignment(
        aligned=True,
        represented_claims=("two-million-dollar carpet",),
        supported_claims=("two-million-dollar carpet",),
        confidence=0.94,
    )
    evaluation = ThumbnailEvaluation(
        accepted=True,
        technical_score=1.0,
        composition_score=0.9,
        readability_score=0.95,
        concept_fit_score=0.9,
        curiosity_score=0.9,
        visual_quality_score=0.9,
        truthfulness_score=1.0,
        promise_alignment=alignment,
    )

    assert constraint.preserve_identity
    assert evaluation.promise_alignment.aligned
    assert RepresentationMode.DOCUMENTARY.value == "documentary"


def test_search_state_tracks_bounded_exploration() -> None:
    budget = SearchBudget(
        max_concepts=4,
        max_asset_plans=8,
        max_attempts=12,
        max_generation_calls=2,
        max_visual_evaluations=4,
        max_runtime_seconds=120.0,
        max_cost=0.50,
    )
    state = SearchState(
        attempted_plan_ids=("plan-1",),
        attempted_strategy_keys=("source_frame",),
        attempts_used=1,
    )

    assert budget.max_attempts == 12
    assert state.attempted_plan_ids == ("plan-1",)


def test_diagnosis_preserves_repair_direction() -> None:
    diagnosis = EvaluationDiagnosis(
        kind=EvaluationFailureKind.ASSET,
        description="The selected frame cannot support the concept.",
        repairable=True,
        recommended_actions=("try another frame", "try subject extraction"),
        confidence=0.88,
    )

    assert diagnosis.repairable
    assert len(diagnosis.recommended_actions) == 2
