from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.packaging.models import ClipPackaging
from src.thumbnails.pipeline_v11 import ThumbnailPipelineV11


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a V1.1 thumbnail from the clean source section."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--clip-id", type=int, required=True)
    args = parser.parse_args()

    project = Path(args.project)
    packaging_path = project / "packaging" / f"clip_{args.clip_id:02d}.json"
    source_path = project / "source" / f"section_{args.clip_id:02d}.mp4"
    output_path = project / "packaging" / f"thumbnail_v11_clip_{args.clip_id:02d}.jpg"

    data = json.loads(packaging_path.read_text(encoding="utf-8"))
    packaging = ClipPackaging(**data)

    print(f"Clean source: {source_path}")
    print(f"Packaging:    {packaging_path}")

    pipeline = ThumbnailPipelineV11()
    result = pipeline.generate(
        packaging=packaging,
        source_video_path=source_path,
        output_path=output_path,
    )

    print()
    print("RESULT")
    print("status:", result.status)
    print("output:", result.output_path)
    print("attempt:", result.selected_attempt_id)
    print("failure:", result.failure_reason)


if __name__ == "__main__":
    main()
