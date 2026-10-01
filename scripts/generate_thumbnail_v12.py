from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.packaging.models import ClipPackaging
from src.thumbnails.pipeline_v12 import ThumbnailPipelineV12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a V1.2 thumbnail with semantic visual intelligence."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--clip-id", type=int, required=True)
    args = parser.parse_args()

    project = Path(args.project)
    packaging_path = project / "packaging" / f"clip_{args.clip_id:02d}.json"
    source_path = project / "source" / f"section_{args.clip_id:02d}.mp4"
    output_path = project / "packaging" / f"thumbnail_v12_clip_{args.clip_id:02d}.jpg"

    data = json.loads(packaging_path.read_text(encoding="utf-8"))
    packaging = ClipPackaging(**data)

    print(f"Clean source: {source_path}")
    print(f"Packaging:    {packaging_path}")

    pipeline = ThumbnailPipelineV12()
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
