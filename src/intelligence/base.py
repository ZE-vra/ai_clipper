"""Module 4 Interface: Abstract AI Director Layer."""

from abc import ABC, abstractmethod
from src.schemas import CandidateManifest, EvaluationManifest


class BaseAIDirector(ABC):

    @abstractmethod
    def evaluate_candidates(
        self, candidate_manifest: CandidateManifest
    ) -> EvaluationManifest:
        """Evaluates candidate windows and returns structured quality scores."""
        pass