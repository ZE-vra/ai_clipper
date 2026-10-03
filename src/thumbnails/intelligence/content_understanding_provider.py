"""Generate V2 thumbnail inputs from a transcript using an explicit AI provider."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, TypeVar

from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import (
    ContentEntity,
    ContentEvent,
    ContentUnderstanding,
)

T = TypeVar("T")


def _call_with_transient_retries(
    request: Callable[[], T],
    *,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry temporary Gemini service failures with exponential backoff.

    Keep quota errors and non-transient failures immediate: retrying those
    requests cannot resolve the underlying problem.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return request()
        except Exception as exc:
            error_text = str(exc).lower()

            if "429" in error_text or "resource_exhausted" in error_text:
                raise RuntimeError(
                    "Gemini quota/rate limit reached. Check the API quota "
                    "and billing for the configured project."
                ) from exc

            transient_error = any(
                marker in error_text
                for marker in (
                    "500",
                    "502",
                    "503",
                    "504",
                    "unavailable",
                    "timeout",
                    "timed out",
                    "connection reset",
                    "connection error",
                )
            )
            if not transient_error:
                raise RuntimeError(
                    f"Gemini content-understanding request failed: {exc}"
                ) from exc

            if attempt == max_retries:
                raise RuntimeError(
                    "Gemini content-understanding request failed after "
                    f"{max_retries} attempts: {exc}"
                ) from exc

            wait_time = 2**attempt
            print(
                "Gemini temporary server error. "
                f"Retrying in {wait_time}s "
                f"(attempt {attempt}/{max_retries})..."
            )
            sleep(wait_time)

    raise RuntimeError("Gemini content-understanding request failed unexpectedly.")


class GeminiContentUnderstandingProvider:
    """Turn transcript text into a validated brief and content understanding.

    The generation callable is injected so parsing and validation remain
    deterministic and can be tested without network access.
    """

    def __init__(self, generate_json: Callable[[str], str]) -> None:
        self._generate_json = generate_json

    @classmethod
    def from_env(cls, model: str = "gemini-3.5-flash") -> "GeminiContentUnderstandingProvider":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Set it to use transcript-based understanding."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The Google GenAI SDK is required for transcript-based understanding."
            ) from exc

        client = genai.Client(api_key=api_key)

        def generate(prompt: str) -> str:
            response = _call_with_transient_retries(
                lambda: client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
            )
            if not response.text:
                raise ValueError("Gemini returned an empty content-understanding response.")
            return response.text.strip()

        return cls(generate)

    def understand(self, transcript: str) -> tuple[ThumbnailBrief, ContentUnderstanding]:
        """Create a concise creative brief and grounded events/entities."""
        if not transcript.strip():
            raise ValueError("Transcript must not be empty.")

        prompt = f"""
You are preparing inputs for a video thumbnail creative system.
Use only information supported by this transcript. Do not invent visual details; put only transcript-grounded facts in entities, events, and claims.
Create a concise, truthful thumbnail brief with a compelling hook, without
misrepresenting what the video says. Thumbnail text must fit a 16:9 YouTube
thumbnail: write both "core_hook" and "promise" as short on-image copy, each
2–5 words and at most 28 characters where possible. Use punchy fragments,
not full sentences or explanations. Preserve the central truthful meaning;
prefer a shorter phrase over adding context. Do not put the curiosity angle,
background explanation, or multiple ideas into the on-image copy.

Return one JSON object with this exact structure:
{{
  "brief": {{
    "core_hook": "short hook",
    "subject": "main subject",
    "promise": "what the viewer will understand or see",
    "curiosity_angle": "truthful open question or curiosity",
    "emotional_direction": "optional direction",
    "important_entities": ["entity ids"],
    "important_objects": [],
    "important_locations": [],
    "visual_evidence": ["visual ideas that could be verified in frames"],
    "constraints": [],
    "forbidden_misrepresentations": ["claims the thumbnail must not imply"]
  }},
  "understanding": {{
    "entities": [
      {{"entity_id":"stable-id","label":"name","kind":"person, place, object, or concept","confidence":0.0,"evidence":["transcript evidence"]}}
    ],
    "events": [
      {{"event_id":"stable-id","start_time":0.0,"end_time":0.0,"description":"event","entity_ids":["stable-id"],"importance":0.0,"confidence":0.0}}
    ],
    "themes": [],
    "claims": [],
    "confidence": 0.0
  }}
}}

Use timestamps only when the transcript includes timestamps; otherwise use
0.0 for both event times. Confidence and importance must be between 0 and 1.
Transcript:
{transcript}
"""
        try:
            data = json.loads(self._generate_json(prompt))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Content-understanding provider returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("brief"), dict) or not isinstance(data.get("understanding"), dict):
            raise ValueError("Provider response must contain 'brief' and 'understanding' objects.")

        brief_data = data["brief"]
        sequence_fields = {
            "important_entities", "important_objects", "important_locations",
            "visual_evidence", "constraints", "forbidden_misrepresentations",
        }
        brief = ThumbnailBrief(**{
            key: tuple(value) if key in sequence_fields else value
            for key, value in brief_data.items()
        })

        understanding_data = data["understanding"]
        understanding = ContentUnderstanding(
            entities=tuple(
                ContentEntity(**{
                    key: tuple(value) if key == "evidence" else value
                    for key, value in item.items()
                })
                for item in understanding_data.get("entities", [])
            ),
            events=tuple(
                ContentEvent(**{
                    key: tuple(value) if key == "entity_ids" else value
                    for key, value in item.items()
                })
                for item in understanding_data.get("events", [])
            ),
            themes=tuple(understanding_data.get("themes", ())),
            claims=tuple(understanding_data.get("claims", ())),
            confidence=understanding_data.get("confidence", 0.0),
        )
        return brief, understanding
