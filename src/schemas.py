"""Data contracts for all modules across the clipping pipeline."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==========================================
# 0. Source Abstraction
# ==========================================
@dataclass
class VideoSource:
    source_type: str  # "youtube" or "local"
    location: str  # URL or absolute path string
    title: Optional[str] = None


# ==========================================
# 1. Perception / Transcription Layer
# ==========================================
@dataclass
class TranscriptSegment:
    id: int
    start: float  # Absolute seconds
    end: float  # Absolute seconds
    text: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class Transcript:
    source: VideoSource
    duration: float
    language: str
    segments: List[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================
# 2. Candidate Discovery Layer
# ==========================================
@dataclass
class CandidateWindow:
    candidate_id: int
    start_time: float
    end_time: float
    transcript_text: str
    segments: List[TranscriptSegment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = self.duration  # Expose property during dict export
        return d


@dataclass
class CandidateManifest:
    source: VideoSource
    total_candidates: int
    candidates: List[CandidateWindow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ==========================================
# 3. Intelligence / AI Evaluation Layer
# ==========================================
@dataclass
class CandidateEvaluation:
    candidate_id: int
    score: float  # 0.0 to 10.0 scale
    reason: str
    suggested_title: Optional[str] = None
    # Gemini may refine the candidate to exact transcript segment boundaries.
    # Optional for backward compatibility with existing evaluation manifests.
    start_segment_id: Optional[int] = None
    end_segment_id: Optional[int] = None


@dataclass
class EvaluationManifest:
    source: VideoSource
    evaluations: List[CandidateEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================
# 4. Planning & Execution Layer
# ==========================================
@dataclass
class ClipDecision:
    clip_id: int
    candidate_id: int
    snapped_start_time: float
    snapped_end_time: float
    final_score: float
    reason: str
    title: Optional[str] = None

    @property
    def duration(self) -> float:
        return round(self.snapped_end_time - self.snapped_start_time, 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = self.duration
        return d


@dataclass
class ClipManifest:
    source: VideoSource
    selected_clips: List[ClipDecision] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

@dataclass
class FinalRenderedClip:
    clip_id: int
    file_path: str
    title: str

    @property
    def exists(self) -> bool:
        return Path(self.file_path).exists()