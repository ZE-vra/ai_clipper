"""Gemini-powered packaging director."""

import json
import os
import random
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.exceptions import ClipperError
from src.schemas import ClipDecision
from src.packaging.base import BasePackager


load_dotenv()


class PackagingError(ClipperError):
    """Raised when clip packaging fails."""


class GeminiPackager(BasePackager):
    """Generate publish-ready metadata with Gemini and fail over safely."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        fallback_model: str = "gemini-3.5-flash",
        max_retries: int = 2,
        initial_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 30.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.api_key:
            raise PackagingError(
                "GEMINI_API_KEY is not configured. "
                "Add it to your .env file."
            )

        if max_retries < 1:
            raise PackagingError(
                "max_retries must be at least 1."
            )

        if initial_backoff_seconds < 0:
            raise PackagingError(
                "initial_backoff_seconds cannot be negative."
            )

        if max_backoff_seconds <= 0:
            raise PackagingError(
                "max_backoff_seconds must be greater than 0."
            )

        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.initial_backoff_seconds = (
            initial_backoff_seconds
        )
        self.max_backoff_seconds = (
            max_backoff_seconds
        )

        try:
            self.client = genai.Client(
                api_key=self.api_key
            )
        except Exception as exc:
            raise PackagingError(
                f"Failed to initialize Gemini client: {exc}"
            ) from exc

    def package_clip(
        self,
        clip: ClipDecision,
        transcript_text: str,
        source_title: str,
    ) -> dict:
        """Generate structured packaging metadata for one selected clip."""

        if not transcript_text.strip():
            raise PackagingError(
                f"Cannot package clip {clip.clip_id}: "
                "transcript context is empty."
            )

        prompt = self._build_prompt(
            clip=clip,
            transcript_text=transcript_text,
            source_title=source_title,
        )

        models = self._model_chain()

        last_error: Optional[Exception] = None

        for model_index, model_name in enumerate(models):
            print(
                f"Packaging clip {clip.clip_id} "
                f"with {model_name}..."
            )

            try:
                return self._generate_with_model(
                    model_name=model_name,
                    prompt=prompt,
                    clip_id=clip.clip_id,
                )

            except PackagingError as exc:
                last_error = exc

                is_last_model = (
                    model_index
                    == len(models) - 1
                )

                if is_last_model:
                    break

                print(
                    f"Primary packaging model "
                    f"{model_name} failed for clip "
                    f"{clip.clip_id}: {exc}"
                )

                print(
                    f"Falling back to "
                    f"{models[model_index + 1]}..."
                )

            except Exception as exc:
                last_error = exc

                is_last_model = (
                    model_index
                    == len(models) - 1
                )

                if is_last_model:
                    break

                print(
                    f"Packaging model "
                    f"{model_name} failed for clip "
                    f"{clip.clip_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"Falling back to "
                    f"{models[model_index + 1]}..."
                )

        raise PackagingError(
            f"Failed to generate packaging for clip "
            f"{clip.clip_id} after trying "
            f"{', '.join(models)}: "
            f"{last_error}"
        ) from last_error

    def _generate_with_model(
        self,
        model_name: str,
        prompt: str,
        clip_id: int,
    ) -> dict:
        """Call one model with bounded exponential-backoff retries."""

        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                response = (
                    self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            response_mime_type=(
                                "application/json"
                            ),
                            response_schema={
                                "type": "OBJECT",
                                "properties": {
                                    "title": {
                                        "type": "STRING",
                                    },
                                    "hook": {
                                        "type": "STRING",
                                    },
                                    "caption": {
                                        "type": "STRING",
                                    },
                                    "description": {
                                        "type": "STRING",
                                    },
                                    "thumbnail_text": {
                                        "type": "STRING",
                                    },
                                    "content_angle": {
                                        "type": "STRING",
                                    },
                                    "hashtags": {
                                        "type": "ARRAY",
                                        "items": {
                                            "type": "STRING",
                                        },
                                    },
                                },
                                "required": [
                                    "title",
                                    "hook",
                                    "caption",
                                    "description",
                                    "thumbnail_text",
                                    "content_angle",
                                    "hashtags",
                                ],
                            },
                            automatic_function_calling=(
                                types.AutomaticFunctionCallingConfig(
                                    disable=True
                                )
                            ),
                        ),
                    )
                )

                if not response.text:
                    raise PackagingError(
                        f"Gemini returned an empty response "
                        f"for clip {clip_id}."
                    )

                try:
                    data = json.loads(
                        response.text
                    )
                except json.JSONDecodeError as exc:
                    raise PackagingError(
                        f"Gemini returned invalid JSON "
                        f"for clip {clip_id}."
                    ) from exc

                self._validate_response(
                    data,
                    clip_id,
                )

                if model_name != self.model:
                    print(
                        f"Packaging succeeded using "
                        f"fallback model {model_name}."
                    )

                return data

            except PackagingError as exc:
                last_error = exc

                if not self._should_retry_packaging_error(
                    exc
                ):
                    break

            except Exception as exc:
                last_error = exc

                if not self._is_retryable_exception(
                    exc
                ):
                    break

            if attempt >= self.max_retries:
                break

            delay = self._calculate_backoff(
                attempt
            )

            print(
                f"Packaging attempt {attempt} failed "
                f"for clip {clip_id} "
                f"using {model_name}: "
                f"{type(last_error).__name__}: "
                f"{last_error}"
            )

            print(
                f"Waiting {delay:.1f}s before retry..."
            )

            time.sleep(delay)

        if last_error is None:
            raise PackagingError(
                f"Model {model_name} failed for "
                f"clip {clip_id} without an error."
            )

        raise PackagingError(
            f"Model {model_name} failed for "
            f"clip {clip_id}: {last_error}"
        ) from last_error

    def _model_chain(self) -> list[str]:
        """Return the primary/fallback model chain without duplicates."""

        models = [
            self.model,
            self.fallback_model,
        ]

        unique_models = []

        for model in models:
            if model and model not in unique_models:
                unique_models.append(model)

        return unique_models

    def _calculate_backoff(
        self,
        attempt: int,
    ) -> float:
        """
        Calculate exponential backoff with jitter.

        attempt=1 -> roughly 2s
        attempt=2 -> roughly 4s
        attempt=3 -> roughly 8s
        """

        exponential_delay = min(
            self.initial_backoff_seconds
            * (2 ** (attempt - 1)),
            self.max_backoff_seconds,
        )

        jitter = random.uniform(
            0.0,
            exponential_delay * 0.25,
        )

        return min(
            exponential_delay + jitter,
            self.max_backoff_seconds,
        )

    @staticmethod
    def _is_retryable_exception(
        exc: Exception,
    ) -> bool:
        """Identify transient Gemini/API failures."""

        message = str(exc).lower()

        retryable_statuses = (
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
        )

        return any(
            marker in message
            for marker in retryable_statuses
        )

    @classmethod
    def _should_retry_packaging_error(
        cls,
        exc: PackagingError,
    ) -> bool:
        """Decide whether a packaging error represents a transient API failure."""

        message = str(exc).lower()

        return any(
            marker in message
            for marker in (
                "503",
                "502",
                "504",
                "429",
                "408",
                "unavailable",
                "timeout",
                "timed out",
            )
        )

    @staticmethod
    def _build_prompt(
        clip: ClipDecision,
        transcript_text: str,
        source_title: str,
    ) -> str:
        return f"""
You are the packaging director for a short-form video content system.

Your job is to understand the actual selected clip and create compelling,
accurate packaging for it.

You are NOT allowed to invent events, quotes, reactions, outcomes,
people, or facts that are not supported by the provided context.

SOURCE VIDEO:
{source_title}

SELECTED CLIP:
Clip ID: {clip.clip_id}
Start: {clip.snapped_start_time:.2f}s
End: {clip.snapped_end_time:.2f}s
Duration: {clip.duration:.2f}s

CURRENT CLIP TITLE:
{clip.title or "None"}

SELECTION REASON:
{clip.reason}

TRANSCRIPT / CLIP CONTEXT:
{transcript_text}

PACKAGING OBJECTIVES:

1. ACCURACY
Base every claim on the supplied clip context.

2. CURIOSITY
Make the viewer want to watch without using dishonest clickbait.

3. SPECIFICITY
Prefer concrete details from the clip over generic hype.

Avoid generic phrases such as:
- "You won't believe this"
- "This changed everything"
- "Wait until you see this"
- "The ending is insane"

unless the actual content specifically justifies them.

4. HUMAN LANGUAGE
Write like an experienced short-form content strategist.
Do not sound corporate, robotic, or overly polished.

5. DO NOT OVER-SPOIL
Where appropriate, create curiosity without immediately
revealing the complete payoff.

6. TITLE
Create one concise, specific title.

7. HOOK
Create a short hook suitable for on-screen text or an opening line.

8. CAPTION
Write a natural social-media caption that adds context or curiosity.

9. DESCRIPTION
Briefly explain what happens in the clip.

10. THUMBNAIL TEXT
Create short thumbnail text.

Prefer 2-6 words.

11. CONTENT ANGLE
Identify the strongest genuine angle of the clip.

Examples:
surprise, generosity, challenge, experiment, conflict,
transformation, humor, reveal, reaction, story, achievement,
curiosity, unexpected outcome.

Use another label if it fits better.

12. HASHTAGS
Generate a small set of relevant hashtags.
Do not spam generic hashtags.

Return ONLY the requested JSON structure.
""".strip()

    @staticmethod
    def _validate_response(
        data: dict,
        clip_id: int,
    ) -> None:
        if not isinstance(data, dict):
            raise PackagingError(
                f"Packaging response for clip {clip_id} "
                "must be a JSON object."
            )

        required_fields = {
            "title",
            "hook",
            "caption",
            "description",
            "thumbnail_text",
            "content_angle",
            "hashtags",
        }

        missing = (
            required_fields
            - data.keys()
        )

        if missing:
            raise PackagingError(
                f"Packaging response for clip {clip_id} "
                f"is missing fields: {sorted(missing)}"
            )

        string_fields = {
            "title",
            "hook",
            "caption",
            "description",
            "thumbnail_text",
            "content_angle",
        }

        for field in string_fields:
            if not isinstance(
                data[field],
                str,
            ):
                raise PackagingError(
                    f"Packaging field '{field}' for clip "
                    f"{clip_id} must be a string."
                )

            if not data[field].strip():
                raise PackagingError(
                    f"Packaging field '{field}' for clip "
                    f"{clip_id} cannot be empty."
                )

        if not isinstance(
            data["hashtags"],
            list,
        ):
            raise PackagingError(
                f"Packaging field 'hashtags' for clip "
                f"{clip_id} must be a list."
            )

        for hashtag in data["hashtags"]:
            if not isinstance(
                hashtag,
                str,
            ):
                raise PackagingError(
                    f"Every hashtag for clip {clip_id} "
                    "must be a string."
                )