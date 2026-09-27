"""Packaging layer interface."""

from abc import ABC, abstractmethod

from src.schemas import ClipDecision


class BasePackager(ABC):
    """Interface for generating publish-ready metadata for a clip."""

    @abstractmethod
    def package_clip(
        self,
        clip: ClipDecision,
        transcript_text: str,
        source_title: str,
    ) -> dict:
        """Generate packaging metadata for one selected clip."""
        raise NotImplementedError