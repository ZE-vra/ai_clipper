"""Command-line entry point for the AI clipping pipeline."""

import sys

from src.exceptions import ClipperError
from src.intelligence.gemini import GeminiDirector
from src.pipeline.orchestrator import PipelineOrchestrator


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python cli.py <youtube_url_or_local_file>")
        return 1

    source_location = sys.argv[1]

    try:
        intelligence_engine = GeminiDirector()

        pipeline = PipelineOrchestrator(
            intelligence_engine=intelligence_engine,
        )

        rendered_clips = pipeline.run(source_location)

    except ClipperError as exc:
        print()
        print("=" * 60)
        print("PIPELINE FAILED")
        print("=" * 60)
        print(exc)
        return 1

    except KeyboardInterrupt:
        print()
        print("Pipeline interrupted.")
        return 130

    except Exception as exc:
        print()
        print("=" * 60)
        print("UNEXPECTED PIPELINE ERROR")
        print("=" * 60)
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    if not rendered_clips:
        print("No clips were selected.")
        return 0

    print(f"Rendered clips: {len(rendered_clips)}")

    for clip in rendered_clips:
        print()
        print(f"Clip {clip.clip_id}")
        print(f"Title: {clip.title}")
        print(f"Path:  {clip.file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())