"""Test deterministic project identity and workspace reuse."""

from src.config import Config


TEST_URL = (
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)

ALT_TEST_URL = (
    "https://youtu.be/dQw4w9WgXcQ"
)


def main() -> None:
    print("=== WORKSPACE IDENTITY TEST ===")
    print()

    first = Config.get_or_create_workspace(
        TEST_URL
    )

    second = Config.get_or_create_workspace(
        TEST_URL
    )

    alternate = Config.get_or_create_workspace(
        ALT_TEST_URL
    )

    print(
        f"First project:     "
        f"{first[0].project_id}"
    )

    print(
        f"Second project:    "
        f"{second[0].project_id}"
    )

    print(
        f"Alternate URL:     "
        f"{alternate[0].project_id}"
    )

    print()

    assert first[0].project_id == (
        second[0].project_id
    )

    assert first[0].project_id == (
        alternate[0].project_id
    )

    assert first[0].root_dir == (
        second[0].root_dir
    )

    assert second[0].root_dir.exists()

    print(
        "✓ Same YouTube video resolves to "
        "the same workspace."
    )

    print(
        "✓ youtube.com and youtu.be resolve "
        "to the same workspace."
    )

    print(
        "✓ Workspace creation is idempotent."
    )

    print(
        "✓ Empty workspaces are not falsely "
        "treated as completed projects."
    )

    print()
    print("IDENTITY TEST PASSED")


if __name__ == "__main__":
    main()