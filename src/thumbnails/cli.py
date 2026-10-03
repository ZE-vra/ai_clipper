"""Dedicated CLI for the V2 thumbnail pipeline; leaves the clipper CLI unchanged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.thumbnails.domain.brief import ThumbnailBrief
from src.thumbnails.domain.content import ContentEntity, ContentEvent, ContentUnderstanding
from src.thumbnails.domain.geometry import BoundingBox, Region, Size
from src.thumbnails.domain.target import ThumbnailTarget
from src.thumbnails.frame_extractor import FFmpegFrameExtractor
from src.thumbnails.orchestration.thumbnail_orchestrator import ThumbnailOrchestrator
from src.thumbnails.perception.frame_discovery import FrameDiscovery
from src.thumbnails.perception.frame_perception_analyzer import FramePerceptionAnalyzer
from src.thumbnails.perception.subjects import SubjectAnalysisEvidence, SubjectEvidence, SubjectKind
from src.thumbnails.perception.subject_analyzer import SubjectAnalyzer
from src.thumbnails.perception.video_frame_sampler import (
    FFprobeVideoDurationReader,
    ThumbnailFrameSampler,
)


class EmptySubjectAnalyzer:
    """Explicit no-model mode: provide no semantic subject claims."""

    def analyze(self, image_path: Path) -> SubjectAnalysisEvidence:
        return SubjectAnalysisEvidence(subjects=())


class LocalModelSubjectAnalyzer:
    """Adapt the configured local visual model to the V2 perception contract."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Subject model file does not exist: {model_path}. "
                "Provide a local checkpoint; the CLI will not download weights."
            )
        from src.thumbnails.perception.local_visual_intelligence import (
            LocalVisualIntelligenceAnalyzer,
        )

        self._analyzer = LocalVisualIntelligenceAnalyzer(model_path=model_path)

    def analyze(self, image_path: Path) -> SubjectAnalysisEvidence:
        result = self._analyzer.analyze(image_path)
        subjects = tuple(
            SubjectEvidence(
                subject_id=subject.subject_id,
                kind=SubjectKind.PERSON,
                confidence=subject.confidence,
                bounds=subject.bounds,
                focal_point=subject.bounds.center,
                prominence=subject.importance,
            )
            for subject in result.subjects
        )
        return SubjectAnalysisEvidence(
            subjects=subjects,
            analysis_version=result.analysis_version,
        )


def _load_request(path: Path) -> tuple[ThumbnailBrief, ContentUnderstanding, ThumbnailTarget]:
    """Load the explicit, structured creative input for one V2 run."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain a top-level object.")

    try:
        brief_data = data["brief"]
        understanding_data = data["understanding"]
    except KeyError as exc:
        raise ValueError(f"Input JSON is missing required section: {exc.args[0]}") from exc

    brief = ThumbnailBrief(
        **{
            key: tuple(value) if key in {
                "important_entities", "important_objects", "important_locations",
                "visual_evidence", "constraints", "forbidden_misrepresentations",
            } else value
            for key, value in brief_data.items()
        }
    )

    understanding = ContentUnderstanding(
        entities=tuple(ContentEntity(**entity) for entity in understanding_data.get("entities", [])),
        events=tuple(
            ContentEvent(
                **{
                    key: tuple(value) if key == "entity_ids" else value
                    for key, value in event.items()
                }
            )
            for event in understanding_data.get("events", [])
        ),
        themes=tuple(understanding_data.get("themes", ())),
        claims=tuple(understanding_data.get("claims", ())),
        confidence=understanding_data.get("confidence", 0.0),
    )

    target_data = data.get("target", {})
    width = int(target_data.get("width", 1280))
    height = int(target_data.get("height", 720))
    safe_regions = tuple(
        Region(
            name=region["name"],
            bounds=BoundingBox(**region["bounds"]),
        )
        for region in target_data.get("safe_regions", [])
    )
    target = ThumbnailTarget(
        target_id=target_data.get("target_id", "youtube-16x9"),
        platform=target_data.get("platform", "youtube"),
        size=Size(width=width, height=height),
        safe_regions=safe_regions,
        minimum_text_size=float(target_data.get("minimum_text_size", 0.0)),
        small_scale_preview_width=int(target_data.get("small_scale_preview_width", 160)),
    )
    return brief, understanding, target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a thumbnail with the V2 creative pipeline."
    )
    parser.add_argument("source_video", type=Path, help="Path to a local source video.")
    parser.add_argument("--input", required=True, type=Path, help="JSON file containing brief and content understanding.")
    parser.add_argument("--output", required=True, type=Path, help="Destination image path.")
    parser.add_argument("--candidates-dir", type=Path, help="Directory for sampled frames and derived assets.")
    parser.add_argument(
        "--subject-model",
        type=Path,
        help="Optional local YOLO segmentation checkpoint. If omitted, semantic subject detection is disabled.",
    )
    parser.add_argument("--samples", type=int, default=9, help="Number of source frames to inspect (default: 9).")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable (default: ffmpeg).")
    parser.add_argument("--ffprobe", default="ffprobe", help="FFprobe executable (default: ffprobe).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.samples < 1:
            raise ValueError("--samples must be at least 1.")
        if not args.source_video.is_file():
            raise FileNotFoundError(f"Source video does not exist: {args.source_video}")
        brief, understanding, target = _load_request(args.input)

        subject_analyzer: SubjectAnalyzer = (
            LocalModelSubjectAnalyzer(args.subject_model)
            if args.subject_model
            else EmptySubjectAnalyzer()
        )
        sampler = ThumbnailFrameSampler(
            extractor=FFmpegFrameExtractor(ffmpeg_binary=args.ffmpeg),
            duration_reader=FFprobeVideoDurationReader(ffprobe_binary=args.ffprobe).read,
            sample_count=args.samples,
        )
        discovery = FrameDiscovery(
            sampler=sampler,
            perception=FramePerceptionAnalyzer(subject_analyzer=subject_analyzer),
        )
        mask_provider = None
        if args.subject_model:
            from ultralytics import YOLO
            from src.thumbnails.perception.ultralytics_subject_mask_provider import (
                UltralyticsSubjectMaskProvider,
            )

            mask_provider = UltralyticsSubjectMaskProvider(YOLO(str(args.subject_model)))

        orchestrator = ThumbnailOrchestrator(
            frame_discovery=discovery,
            subject_mask_provider=mask_provider,
        )
        candidates_dir = args.candidates_dir or args.output.parent / f"{args.output.stem}_work"
        result = orchestrator.generate(
            source_video_path=args.source_video,
            output_path=args.output,
            candidates_dir=candidates_dir,
            brief=brief,
            understanding=understanding,
            target=target,
        )
        if result.output_path:
            print(f"Thumbnail {result.status.value}: {result.output_path}")
            print(f"Selected attempt: {result.selected_attempt_id}")
        else:
            print(f"Thumbnail {result.status.value}: {result.failure_reason}", file=sys.stderr)
        return 0 if result.status.value in {"success", "fallback_success"} else 1
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Thumbnail V2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
