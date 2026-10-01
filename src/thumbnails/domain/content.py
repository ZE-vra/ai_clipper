from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContentEntity:
    """A semantically meaningful entity identified in the source content."""

    entity_id: str
    label: str
    kind: str
    confidence: float
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must not be blank.")
        if not self.label.strip():
            raise ValueError("label must not be blank.")
        if not self.kind.strip():
            raise ValueError("kind must not be blank.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True)
class ContentEvent:
    """A temporally grounded event or moment in the source content."""

    event_id: str
    start_time: float
    end_time: float
    description: str
    entity_ids: tuple[str, ...] = ()
    importance: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be blank.")
        if self.start_time < 0:
            raise ValueError("start_time must not be negative.")
        if self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time.")
        if not self.description.strip():
            raise ValueError("description must not be blank.")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0 and 1.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True)
class ContentUnderstanding:
    """Structured understanding of what happens in a clip."""

    entities: tuple[ContentEntity, ...]
    events: tuple[ContentEvent, ...]
    themes: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
