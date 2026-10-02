from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.thumbnails.domain.evaluation import PromiseAlignment, ThumbnailEvaluation
from src.thumbnails.domain.plans import ThumbnailRenderPlan
from src.thumbnails.domain.target import ThumbnailTarget


class DeterministicThumbnailEvaluator:
    """Validates properties that can be established without visual judgment.

    This evaluator deliberately does not pretend to measure clickability or
    creative quality. Those belong behind the later visual/content evaluators.
    """

    def evaluate(
        self,
        *,
        output_path: str | Path,
        plan: ThumbnailRenderPlan,
        target: ThumbnailTarget,
    ) -> ThumbnailEvaluation:
        path = Path(output_path)
        failures: list[str] = []

        if not path.is_file():
            failures.append("rendered thumbnail does not exist")
            return self._result(failures, 0.0, 0.0, 0.0)

        try:
            with Image.open(path) as image:
                width, height = image.size
                if (width, height) != (target.size.width, target.size.height):
                    failures.append(
                        f"rendered size {width}x{height} does not match "
                        f"target {target.size.width}x{target.size.height}"
                    )
                if image.width <= 0 or image.height <= 0:
                    failures.append("rendered thumbnail has invalid dimensions")
        except (OSError, ValueError) as exc:
            failures.append(f"rendered thumbnail cannot be opened: {exc}")
            return self._result(failures, 0.0, 0.0, 0.0)

        composition_score = 1.0 if plan.composition.visual_placements else 0.0
        readability_score = (
            1.0 if plan.typography.blocks else 0.0
        )

        technical_score = 1.0 if not failures else 0.0
        accepted = not failures and composition_score > 0.0 and readability_score > 0.0

        return ThumbnailEvaluation(
            accepted=accepted,
            technical_score=technical_score,
            composition_score=composition_score,
            readability_score=readability_score,
            concept_fit_score=0.0,
            curiosity_score=0.0,
            visual_quality_score=0.0,
            truthfulness_score=1.0,
            promise_alignment=PromiseAlignment(
                aligned=True,
                rationale="No generative or semantic claim was introduced by this deterministic pass.",
            ),
            hard_failures=tuple(failures),
            rationale=(
                "Deterministic artifact validation passed."
                if accepted
                else "Deterministic artifact validation failed."
            ),
        )

    @staticmethod
    def _result(
        failures: list[str],
        composition_score: float,
        readability_score: float,
        technical_score: float,
    ) -> ThumbnailEvaluation:
        return ThumbnailEvaluation(
            accepted=False,
            technical_score=technical_score,
            composition_score=composition_score,
            readability_score=readability_score,
            truthfulness_score=1.0,
            promise_alignment=PromiseAlignment(
                aligned=True,
                rationale="No generative or semantic claim was introduced by this deterministic pass.",
            ),
            hard_failures=tuple(failures),
            rationale="Deterministic artifact validation failed.",
        )
