"""Module 4: Gemini Intelligence Provider.

Evaluates candidate windows using Google's Gemini API with:
- Batched candidate evaluation
- Quota-aware rate-limit handling
- Retries for temporary server failures
- Structured JSON output
"""

import json
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.exceptions import IntelligenceError, RateLimitError
from src.intelligence.base import BaseAIDirector
from src.schemas import (
    CandidateEvaluation,
    CandidateManifest,
    CandidateWindow,
    EvaluationManifest,
)

load_dotenv()


class GeminiDirector(BaseAIDirector):
    """Gemini implementation of the AI Director interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        batch_size: int = 40,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise IntelligenceError(
                "GEMINI_API_KEY is not configured in .env or passed explicitly."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _call_gemini(self, prompt: str) -> str:
        """Send a prompt to Gemini with retry handling."""

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    
                    ),
                )

                if not response.text:
                    raise IntelligenceError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except Exception as exc:
                error_text = str(exc).lower()

                if "429" in error_text or "resource_exhausted" in error_text:
                    raise RateLimitError(
                        "Gemini quota/rate limit reached. "
                        "The API reported that the current request quota "
                        "has been exhausted."
                    ) from exc

                transient_error = any(
                    marker in error_text
                    for marker in [
                        "500",
                        "502",
                        "503",
                        "504",
                        "unavailable",
                        "timeout",
                        "timed out",
                        "connection reset",
                        "connection error",
                    ]
                )

                if not transient_error:
                    raise IntelligenceError(
                        f"Gemini request failed: {exc}"
                    ) from exc

                if attempt == self.max_retries:
                    raise IntelligenceError(
                        f"Gemini request failed after "
                        f"{self.max_retries} attempts: {exc}"
                    ) from exc

                wait_time = 2 ** attempt

                print(
                    f"\n⚠️ Gemini temporary server error. "
                    f"Retrying in {wait_time}s "
                    f"(attempt {attempt}/{self.max_retries})..."
                )

                time.sleep(wait_time)

        raise IntelligenceError(
            "Gemini request failed unexpectedly."
        )

    def _evaluate_batch(
        self,
        candidates_batch: List[CandidateWindow],
        video_title: str,
    ) -> List[CandidateEvaluation]:
        """Evaluate one batch of candidate windows."""

        candidates_payload = [
            {
                "candidate_id": candidate.candidate_id,
                "duration_seconds": round(candidate.duration, 1),
                "segments": [
                    {
                        "segment_id": segment.id,
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "text": segment.text,
                    }
                    for segment in candidate.segments
                ],
            }
            for candidate in candidates_batch
        ]

        prompt = f"""
You are an expert video editor and content director.

Your job is to identify genuinely compelling moments from a long-form
video that could stand alone as extracted clips.

A strong candidate should:

1. Make sense on its own.
2. Have a clear beginning and satisfying payoff.
3. End immediately after the meaningful payoff or completed thought.
4. Avoid carrying dead air, setup, repetition, or post-payoff material.
5. Contain something worth watching, such as:
   - humor
   - surprise
   - emotion
   - insight
   - conflict
   - an interesting story
   - a strong opinion
   - an unusual or memorable moment
4. Require as little outside context as possible.
5. Feel natural as a standalone excerpt.

Do NOT favor a candidate merely because it is short.
Do NOT force every clip into a fixed duration.
Do NOT optimize solely for "virality".

Evaluate EVERY candidate from 0.0 to 10.0.

Video title:
"{video_title}"

Candidates:

{json.dumps(candidates_payload, indent=2)}

Return a JSON array containing exactly one evaluation object
for every candidate.

Each object must contain:

- "candidate_id": integer
- "score": number from 0.0 to 10.0
- "reason": short 1-2 sentence explanation
- "suggested_title": string containing a concise 3-6 word title,
  or null if no useful title can be suggested
- "start_segment_id": integer ID of the first segment to include
- "end_segment_id": integer ID of the last segment to include

Boundary rules:
- Both segment IDs MUST come from the candidate's supplied segments.
- Start at the earliest segment needed to understand the moment.
- End at the first segment that completes the payoff, answer, reveal,
  punchline, or meaningful thought.
- Do NOT keep extra material after the payoff.
- Do NOT cut off the payoff mid-sentence.
- Do NOT choose boundaries based on a preferred clip length.
"""

        raw_json = self._call_gemini(prompt)

        try:
            evaluations_raw = json.loads(raw_json)

        except json.JSONDecodeError as exc:
            raise IntelligenceError(
                f"Gemini returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(evaluations_raw, list):
            raise IntelligenceError(
                "Gemini response must be a JSON array."
            )

        evaluations: List[CandidateEvaluation] = []

        for item in evaluations_raw:
            try:
                evaluation = CandidateEvaluation(
                    candidate_id=int(item["candidate_id"]),
                    score=float(item["score"]),
                    reason=str(item.get("reason", "")),
                    suggested_title=item.get("suggested_title"),
                    start_segment_id=(
                        int(item["start_segment_id"])
                        if item.get("start_segment_id") is not None
                        else None
                    ),
                    end_segment_id=(
                        int(item["end_segment_id"])
                        if item.get("end_segment_id") is not None
                        else None
                    ),
                )

                if not 0.0 <= evaluation.score <= 10.0:
                    raise ValueError(
                        f"Score outside valid range: {evaluation.score}"
                    )

                valid_segment_ids = {
                    segment.id
                    for candidate in candidates_batch
                    if candidate.candidate_id == evaluation.candidate_id
                    for segment in candidate.segments
                }

                if (
                    evaluation.start_segment_id is not None
                    and evaluation.start_segment_id not in valid_segment_ids
                ):
                    raise ValueError(
                        "Gemini returned an invalid start_segment_id."
                    )

                if (
                    evaluation.end_segment_id is not None
                    and evaluation.end_segment_id not in valid_segment_ids
                ):
                    raise ValueError(
                        "Gemini returned an invalid end_segment_id."
                    )

                if (
                    evaluation.start_segment_id is not None
                    and evaluation.end_segment_id is not None
                ):
                    candidate_for_evaluation = next(
                        candidate
                        for candidate in candidates_batch
                        if candidate.candidate_id == evaluation.candidate_id
                    )
                    segment_positions = {
                        segment.id: index
                        for index, segment
                        in enumerate(candidate_for_evaluation.segments)
                    }
                    if (
                        segment_positions[evaluation.start_segment_id]
                        > segment_positions[evaluation.end_segment_id]
                    ):
                        raise ValueError(
                            "Gemini returned reversed clip boundaries."
                        )

                evaluations.append(evaluation)

            except (KeyError, TypeError, ValueError) as exc:
                raise IntelligenceError(
                    f"Malformed candidate evaluation: {item}"
                ) from exc

        expected_ids = {
            candidate.candidate_id
            for candidate in candidates_batch
        }

        returned_ids = {
            evaluation.candidate_id
            for evaluation in evaluations
        }

        missing_ids = expected_ids - returned_ids

        if missing_ids:
            raise IntelligenceError(
                "Gemini did not evaluate every candidate. "
                f"Missing candidate IDs: {sorted(missing_ids)}"
            )

        return evaluations

    def evaluate_candidates(
        self,
        candidate_manifest: CandidateManifest,
    ) -> EvaluationManifest:
        """Evaluate all candidate windows in batches."""

        all_candidates = candidate_manifest.candidates

        if not all_candidates:
            return EvaluationManifest(
                source=candidate_manifest.source,
                evaluations=[],
            )

        video_title = candidate_manifest.source.title or "Unknown"

        all_evaluations: List[CandidateEvaluation] = []

        total_batches = (
            len(all_candidates) + self.batch_size - 1
        ) // self.batch_size

        for i in range(0, len(all_candidates), self.batch_size):
            batch = all_candidates[
                i : i + self.batch_size
            ]

            batch_number = (i // self.batch_size) + 1

            print(
                f"  └ Processing batch "
                f"{batch_number}/{total_batches} "
                f"({len(batch)} candidates)..."
            )

            try:
                batch_evaluations = self._evaluate_batch(
                    batch,
                    video_title,
                )

            except RateLimitError:
                print("\n🛑 Gemini quota/rate limit reached.")
                print("    No further batches will be sent.")
                raise

            all_evaluations.extend(batch_evaluations)

            if batch_number < total_batches:
                time.sleep(1.5)

        return EvaluationManifest(
            source=candidate_manifest.source,
            evaluations=all_evaluations,
        )