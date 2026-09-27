from pathlib import Path

from src.rendering.source_provider import (
    SourceAcquisitionError,
    YouTubeSourceProvider,
)


VIDEO_URL = "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"

OUTPUT_PATH = Path(
    "projects/youtube_vp5sSqyZ5Go/source/test_section.mp4"
)

START_TIME = 486.48
END_TIME = 496.48


def main() -> None:
    provider = YouTubeSourceProvider()

    print("=" * 60)
    print("SOURCE PROVIDER TEST")
    print("=" * 60)

    print(f"Source: {VIDEO_URL}")
    print(
        f"Range:  {START_TIME:.2f}s → "
        f"{END_TIME:.2f}s"
    )
    print(f"Output: {OUTPUT_PATH}")
    print()

    try:
        result = provider.acquire_section(
            source_url=VIDEO_URL,
            output_path=OUTPUT_PATH,
            start_time=START_TIME,
            end_time=END_TIME,
        )

    except SourceAcquisitionError as exc:
        print()
        print("SOURCE ACQUISITION FAILED")
        print("-" * 60)
        print(exc)
        return

    print()
    print("=" * 60)
    print("SOURCE ACQUISITION SUCCEEDED")
    print("=" * 60)
    print(f"File: {result}")
    print(f"Size: {result.stat().st_size:,} bytes")
    print()
    print("The section passed full audio/video validation.")


if __name__ == "__main__":
    main()