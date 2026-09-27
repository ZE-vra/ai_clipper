from pathlib import Path
import yt_dlp


YOUTUBE_URL = "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"


def main():
    print("=" * 60)
    print("YOUTUBE ACCESS DIAGNOSTIC")
    print("=" * 60)

    print()
    print(f"URL: {YOUTUBE_URL}")

    print()
    print("yt-dlp version:")
    print(yt_dlp.version.__version__)

    print()
    print("=" * 60)
    print("TESTING METADATA ACCESS")
    print("=" * 60)

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,
        "skip_download": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                YOUTUBE_URL,
                download=False,
            )

    except Exception as exc:
        print()
        print("=" * 60)
        print("YOUTUBE ACCESS FAILED")
        print("=" * 60)
        print()
        print(type(exc).__name__)
        print(str(exc))

        print()
        print(
            "This test only checks whether yt-dlp can access "
            "the YouTube video metadata."
        )
        print(
            "No video or audio was downloaded."
        )

        raise SystemExit(1)

    print()
    print("=" * 60)
    print("YOUTUBE ACCESS SUCCEEDED")
    print("=" * 60)

    print()
    print(f"Title:    {info.get('title')}")
    print(f"Duration: {info.get('duration')} seconds")
    print(f"Video ID: {info.get('id')}")
    print(f"Uploader: {info.get('uploader')}")

    print()
    print("Available formats:", len(info.get("formats", [])))

    print()
    print("Metadata access test PASSED.")


if __name__ == "__main__":
    main()