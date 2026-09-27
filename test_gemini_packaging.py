from src.config import Config
from src.persistence.manifests import (
    load_clip_manifest,
    load_source,
    load_transcript,
)
from src.packaging.gemini_packager import GeminiPackager


SOURCE_URL = "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"


def main() -> None:
    print("=" * 60)
    print("GEMINI PACKAGING TEST")
    print("=" * 60)

    workspace = Config.find_workspace(SOURCE_URL)

    if workspace is None:
        raise RuntimeError(
            "Existing YouTube project was not found."
        )

    print(f"Workspace: {workspace.root_dir}")

    source = load_source(workspace)
    transcript = load_transcript(workspace)
    clip_manifest = load_clip_manifest(workspace)

    if not clip_manifest.selected_clips:
        raise RuntimeError(
            "No selected clips found in the clip plan."
        )

    # Test the first selected clip only.
    clip = clip_manifest.selected_clips[0]

    print()
    print(f"Testing clip: {clip.clip_id}")
    print(f"Title: {clip.title}")
    print(
        f"Range: "
        f"{clip.snapped_start_time:.2f}s → "
        f"{clip.snapped_end_time:.2f}s"
    )
    print(f"Duration: {clip.duration:.2f}s")

    # Collect transcript segments that overlap the selected clip.
    relevant_segments = [
        segment
        for segment in transcript.segments
        if segment.end > clip.snapped_start_time
        and segment.start < clip.snapped_end_time
    ]

    transcript_text = " ".join(
        segment.text.strip()
        for segment in relevant_segments
        if segment.text.strip()
    )

    if not transcript_text:
        raise RuntimeError(
            "Could not find transcript context for the selected clip."
        )

    print()
    print("Transcript context:")
    print("-" * 60)
    print(transcript_text)
    print("-" * 60)

    print()
    print("Generating packaging...")

    packager = GeminiPackager()

    packaging = packager.package_clip(
        clip=clip,
        transcript_text=transcript_text,
        source_title=source.title or "Unknown source",
    )

    print()
    print("=" * 60)
    print("PACKAGING RESULT")
    print("=" * 60)

    print(f"Title:           {packaging['title']}")
    print(f"Hook:            {packaging['hook']}")
    print(f"Caption:         {packaging['caption']}")
    print(f"Description:     {packaging['description']}")
    print(f"Thumbnail text:  {packaging['thumbnail_text']}")
    print(f"Content angle:   {packaging['content_angle']}")

    print("Hashtags:")
    for hashtag in packaging["hashtags"]:
        print(f"  {hashtag}")

    print()
    print("=" * 60)
    print("GEMINI PACKAGING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()