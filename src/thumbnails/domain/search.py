from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBudget:
    """Hard limits for bounded creative search."""

    max_concepts: int
    max_asset_plans: int
    max_attempts: int
    max_generation_calls: int
    max_visual_evaluations: int
    max_runtime_seconds: float
    max_cost: float

    def __post_init__(self) -> None:
        values = (
            ("max_concepts", self.max_concepts),
            ("max_asset_plans", self.max_asset_plans),
            ("max_attempts", self.max_attempts),
            ("max_generation_calls", self.max_generation_calls),
            ("max_visual_evaluations", self.max_visual_evaluations),
        )
        for name, value in values:
            if value < 0:
                raise ValueError(f"{name} must not be negative.")
        if self.max_runtime_seconds < 0:
            raise ValueError("max_runtime_seconds must not be negative.")
        if self.max_cost < 0:
            raise ValueError("max_cost must not be negative.")


@dataclass(frozen=True)
class SearchState:
    """Persistable state of the bounded thumbnail search."""

    attempted_plan_ids: tuple[str, ...] = ()
    attempted_strategy_keys: tuple[str, ...] = ()
    attempted_concept_ids: tuple[str, ...] = ()
    best_attempt_id: str | None = None
    attempts_used: int = 0
    generation_calls_used: int = 0
    evaluation_calls_used: int = 0
    estimated_cost: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("attempts_used", self.attempts_used),
            ("generation_calls_used", self.generation_calls_used),
            ("evaluation_calls_used", self.evaluation_calls_used),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative.")
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must not be negative.")
