"""Regenerate one project's Shorts thumbnail without running the clipper pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.packaging.persistence import load_clip_packaging
from src.thumbnails.domain.results import ThumbnailResultStatus
from src.thumbnails.pipeline_stage import ThumbnailV2Stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate one Shorts thumbnail from saved project artifacts. "
            "Does not rerun transcription, clip selection, rendering, or packaging."
        )
    )
    parser.add_argument(
        "source",
        help="The same local video path or YouTube URL used for the original project.",
    )
    parser.add_argument(
        "--clip-id",
        required=True,
        type=int,
        help="Clip number to regenerate, e.g. --clip-id 1.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=9,
        help="Number of source frames to inspect (default: 9).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path; defaults to the project's thumbnail checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.clip_id < 1:
            raise ValueError("--clip-id must be at least 1.")
        if args.samples < 1:
            raise ValueError("--samples must be at least 1.")

        workspace = Config.find_workspace(args.source)
        if workspace is None:
            raise FileNotFoundError(
                "No saved project was found for this source. Run the main clipper "
                "once first so its project checkpoints exist."
            )

        packaging = load_clip_packaging(workspace, args.clip_id)
        source_section = (
            workspace.source_dir / f"section_{args.clip_id:02d}.mp4"
        )
        if not source_section.is_file():
            raise FileNotFoundError(
                f"Saved source section is missing: {source_section}. "
                "This command will not rerun the full pipeline to recreate it."
            )

        output_path = args.output or (
            workspace.root_dir
            / "thumbnails"
            / f"clip_{args.clip_id:02d}_thumbnail.jpg"
        )
        candidates_dir = (
            workspace.root_dir
            / "thumbnail_work"
            / f"clip_{args.clip_id:02d}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_dir.mkdir(parents=True, exist_ok=True)

        # Deliberately call the stage directly: do not invoke PipelineOrchestrator.
        result = ThumbnailV2Stage(sample_count=args.samples).generate(
            source_video_path=source_section,
            packaging=packaging,
            output_path=output_path,
            candidates_dir=candidates_dir,
        )
        if (
            result.status in {
                ThumbnailResultStatus.SUCCESS,
                ThumbnailResultStatus.FALLBACK_SUCCESS,
            }
            and result.output_path
            and Path(result.output_path).is_file()
        ):
            print(f"Thumbnail {result.status.value}: {result.output_path}")
            print("Reused saved packaging and source section; no AI API calls made.")
            return 0

        print(
            f"Thumbnail {result.status.value}: "
            f"{result.failure_reason or 'no output image produced'}",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Thumbnail-only run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
