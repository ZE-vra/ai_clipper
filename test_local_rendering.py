from pathlib import Path

from src.rendering.ffmpeg_renderer import (
    FFmpegRenderer,
    RenderingError,
)


SOURCE_PATH = Path(
    "projects/youtube_vp5sSqyZ5Go/source/local_provider_test.mp4"
)

OUTPUT_PATH = Path(
    "projects/youtube_vp5sSqyZ5Go/clips/local_provider_render_test.mp4"
)

DURATION = 10.0


def main() -> None:
    renderer = FFmpegRenderer()

    print("=" * 60)
    print("LOCAL RENDERER TEST")
    print("=" * 60)

    print(f"Source: {SOURCE_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Duration: {DURATION:.2f}s")
    print()

    try:
        result = renderer.render_clip(
            source_path=SOURCE_PATH,
            output_path=OUTPUT_PATH,
            start_time=0.0,
            end_time=DURATION,
        )

    except RenderingError as exc:
        print()
        print("RENDERING FAILED")
        print("-" * 60)
        print(exc)
        return

    print()
    print("=" * 60)
    print("RENDERING SUCCEEDED")
    print("=" * 60)
    print(f"File: {result}")
    print(f"Size: {result.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()