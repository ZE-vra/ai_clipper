from src.thumbnails.domain.concepts import CopyRole
from src.thumbnails.layout.v11_copy import V11CopyDirector


def test_money_copy_gets_payoff_and_hook() -> None:
    concept = V11CopyDirector().create(text="$500,000 Jet Surprise")

    assert [block.role for block in concept.blocks] == [
        CopyRole.PAYOFF,
        CopyRole.HOOK,
    ]
    assert concept.blocks[0].text == "$500,000"
    assert concept.blocks[1].text == "Jet Surprise"


def test_plain_short_copy_is_kept_compact() -> None:
    concept = V11CopyDirector().create(text="FLYING FOR $1")

    assert concept.blocks[0].text == "$1"
    assert concept.blocks[1].text == "FLYING FOR"


def test_blank_copy_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        V11CopyDirector().create(text="   ")
