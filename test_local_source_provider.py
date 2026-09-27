from pathlib import Path

from src.rendering.source_provider import (
    LocalSourceProvider,
    SourceAcquisitionError,
)


SOURCE_PATH = Path(
    r"C:\Users\Admin\Videos\videos\b9cc7a86cb57a79164acc05df3a75f16_1790343864860.mp4"
)

OUTPUT_PATH = Path(
    "projects/youtube_vp5sSqyZ5Go/source/local_provider_test.mp4"
)

START_TIME = 100.0
END_TIME = 110.0


def main() -> None:
    provider = LocalSourceProvider()

    print("=" * 60)
    print("LOCAL SOURCE PROVIDER TEST")
    print("=" * 60)

    print(f"Source: {SOURCE_PATH}")
    print(
        f"Range:  {START_TIME:.2f}s → "
        f"{END_TIME:.2f}s"
    )
    print(f"Output: {OUTPUT_PATH}")
    print()

    try:
        result = provider.acquire_section(
            source_path=SOURCE_PATH,
            output_path=OUTPUT_PATH,
            start_time=START_TIME,
            end_time=END_TIME,
        )

    except SourceAcquisitionError as exc:
        print()
        print("LOCAL SOURCE ACQUISITION FAILED")
        print("-" * 60)
        print(exc)
        return

    print()
    print("=" * 60)
    print("LOCAL SOURCE ACQUISITION SUCCEEDED")
    print("=" * 60)
    print(f"File: {result}")
    print(f"Size: {result.stat().st_size:,} bytes")
    print()
    print("The section passed full audio/video validation.")


if __name__ == "__main__":
    main()