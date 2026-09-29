from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.packaging.models import ClipPackaging
from src.thumbnails.frame_extractor import FFmpegFrameExtractor
from src.thumbnails.thumbnail_planner import ThumbnailPlanner
from src.thumbnails.thumbnail_renderer import ThumbnailRenderer


PROJECT = Path("projects/youtube_1WEAJ-DFkHE")
CLIP_ID = 1

SOURCE = PROJECT / "source" / "section_01.mp4"
PACKAGING = PROJECT / "packaging" / "clip_01.json"
OUTPUT = Path("debug/visual/thumbnail_clip_01.jpg")


def get_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return float(result.stdout.strip())


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source section does not exist: {SOURCE}"
        )

    if not PACKAGING.exists():
        raise FileNotFoundError(
            f"Packaging file does not exist: {PACKAGING}"
        )

    packaging_data = json.loads(
        PACKAGING.read_text(encoding="utf-8")
    )

    packaging = ClipPackaging(**packaging_data)

    duration = get_duration(SOURCE)

    planner = ThumbnailPlanner()

    plan = planner.create_plan(
        packaging=packaging,
        duration=duration,
    )

    print(f"Clip: {CLIP_ID}")
    print(f"Duration: {duration:.2f}s")
    print(
        f"Selected frame: "
        f"{plan.frame_timestamp:.2f}s"
    )
    print(f"Thumbnail text: {plan.text}")

    frame_path = (
        Path("debug/visual")
        / "thumbnail_source_frame.jpg"
    )

    extractor = FFmpegFrameExtractor()

    extractor.extract(
        source_path=SOURCE,
        timestamp=plan.frame_timestamp,
        output_path=frame_path,
    )

    renderer = ThumbnailRenderer()

    renderer.render(
        plan=plan,
        source_frame_path=frame_path,
        output_path=OUTPUT,
    )

    print()
    print("THUMBNAIL PREVIEW COMPLETE")
    print(f"Frame: {frame_path}")
    print(f"Thumbnail: {OUTPUT}")


if __name__ == "__main__":
    main()