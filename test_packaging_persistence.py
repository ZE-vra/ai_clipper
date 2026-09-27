from src.config import Config
from src.packaging.models import ClipPackaging
from src.packaging.persistence import (
    load_clip_packaging,
    save_clip_packaging,
)


SOURCE_URL = "https://youtu.be/vp5sSqyZ5Go?si=Q5urbvUO-p64nCnD"


def main() -> None:
    print("=" * 60)
    print("PACKAGING PERSISTENCE TEST")
    print("=" * 60)

    workspace = Config.find_workspace(SOURCE_URL)

    if workspace is None:
        raise RuntimeError(
            "Existing YouTube project was not found."
        )

    print(f"Workspace: {workspace.root_dir}")

    packaging = ClipPackaging(
        clip_id=1,
        title="Replacing My Brother's House After The Prank",
        hook="We couldn't just leave his place ruined...",
        caption=(
            "Slime was way more damaging than sticky notes, "
            "so we had to make it right."
        ),
        description=(
            "The team shocks CJ by revealing they bought "
            "him a brand new house."
        ),
        thumbnail_text="WE BOUGHT HIM A HOUSE",
        content_angle="surprise",
        hashtags=[
            "#surprisereveal",
            "#newhouse",
            "#prankpayoff",
            "#family",
        ],
    )

    print()
    print("Saving packaging...")

    saved_path = save_clip_packaging(
        packaging,
        workspace,
    )

    print(f"Saved: {saved_path}")
    print(f"Exists: {saved_path.exists()}")

    print()
    print("Loading packaging...")

    loaded = load_clip_packaging(
        workspace,
        clip_id=1,
    )

    print(f"Clip ID: {loaded.clip_id}")
    print(f"Title: {loaded.title}")
    print(f"Hook: {loaded.hook}")
    print(f"Thumbnail: {loaded.thumbnail_text}")
    print(f"Angle: {loaded.content_angle}")
    print(f"Hashtags: {loaded.hashtags}")

    assert loaded.clip_id == packaging.clip_id
    assert loaded.title == packaging.title
    assert loaded.hook == packaging.hook
    assert loaded.caption == packaging.caption
    assert loaded.description == packaging.description
    assert (
        loaded.thumbnail_text
        == packaging.thumbnail_text
    )
    assert (
        loaded.content_angle
        == packaging.content_angle
    )
    assert loaded.hashtags == packaging.hashtags

    print()
    print("=" * 60)
    print("PACKAGING PERSISTENCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()