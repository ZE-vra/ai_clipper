from pathlib import Path
import shutil

from src.config import Config
from src.rendering.source_provider import YouTubeSourceProvider


YOUTUBE_URL = "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"


def main():
    print("=" * 60)
    print("YOUTUBE SECTION ACQUISITION TEST")
    print("=" * 60)

    workspace = Config.create_workspace(
        "youtube_section_test_" + "vp5sSqyZ5Go"
    )

    output_path = workspace.source_dir / "test_section.mp4"

    if output_path.exists():
        print()
        print("Removing previous test section...")
        output_path.unlink()

    provider = YouTubeSourceProvider()

    start_time = 0.0
    end_time = 20.0

    print()
    print(f"URL:   {YOUTUBE_URL}")
    print(f"Range: {start_time:.2f}s → {end_time:.2f}s")
    print(f"Output: {output_path}")

    print()
    print("=" * 60)
    print("ACQUIRING YOUTUBE SECTION")
    print("=" * 60)

    try:
        result = provider.acquire_section(
            source_url=YOUTUBE_URL,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        print()
        print("=" * 60)
        print("YOUTUBE SECTION ACQUISITION FAILED")
        print("=" * 60)

        print()
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")

        raise SystemExit(1)

    print()
    print("=" * 60)
    print("ACQUISITION SUCCEEDED")
    print("=" * 60)

    print()
    print(f"Result: {result}")
    print(f"Exists: {result.exists}")

    if not result.exists:
        raise RuntimeError(
            "Acquisition reported success, but output file does not exist."
        )

    size = result.stat().st_size

    print(f"Size:   {size:,} bytes")

    if size <= 0:
        raise RuntimeError(
            "Acquired section is empty."
        )

    print()
    print("=" * 60)
    print("VALIDATING ACQUIRED MEDIA")
    print("=" * 60)

    try:
        provider.validate_media(result)
    except Exception as exc:
        print()
        print("=" * 60)
        print("MEDIA VALIDATION FAILED")
        print("=" * 60)

        print()
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")

        raise SystemExit(1)

    print()
    print("Media validation PASSED.")

    print()
    print("=" * 60)
    print("YOUTUBE SECTION TEST PASSED")
    print("=" * 60)

    print()
    print("Verified:")
    print("- YouTube section acquisition")
    print("- Requested 0–20 second range")
    print("- MP4 output exists")
    print("- Output contains data")
    print("- FFmpeg can decode the acquired media")


if __name__ == "__main__":
    main()