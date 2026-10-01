from __future__ import annotations

import re

from src.thumbnails.domain.concepts import CopyBlock, CopyConcept, CopyRole


_MONEY_RE = re.compile(r"(?<!\w)(\$\s?\d[\d,.]*(?:[KMB])?)(?!\w)", re.IGNORECASE)


class V11CopyDirector:
    """Deterministic thumbnail-copy structuring for the V1.1 vertical slice."""

    def create(self, *, text: str, concept_id: str = "v11-copy") -> CopyConcept:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("thumbnail text must not be blank.")

        match = _MONEY_RE.search(cleaned)

        if match:
            payoff = match.group(1).replace(" ", "")
            remainder = (cleaned[:match.start()] + cleaned[match.end():]).strip(" -:|")
            words = remainder.split()

            if words:
                hook = " ".join(words)
                blocks = (
                    CopyBlock(payoff, CopyRole.PAYOFF, emphasis=1.0),
                    CopyBlock(hook, CopyRole.HOOK, emphasis=0.92),
                )
            else:
                blocks = (CopyBlock(payoff, CopyRole.PAYOFF, emphasis=1.0),)
        else:
            words = cleaned.split()
            if len(words) <= 3:
                blocks = (CopyBlock(cleaned.upper(), CopyRole.HOOK),)
            else:
                midpoint = max(1, len(words) // 2)
                blocks = (
                    CopyBlock(" ".join(words[:midpoint]).upper(), CopyRole.HOOK),
                    CopyBlock(" ".join(words[midpoint:]).upper(), CopyRole.PAYOFF, emphasis=0.96),
                )

        return CopyConcept(
            concept_id=concept_id,
            blocks=blocks,
            rationale="Structured existing packaging copy into a high-hierarchy vertical layout.",
        )
