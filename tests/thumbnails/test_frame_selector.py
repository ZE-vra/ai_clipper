import pytest

from src.thumbnails.frame_selector import (
    DeterministicFrameSelector,
)


def test_default_positions_generate_six_candidates() -> None:
    selector = DeterministicFrameSelector()

    candidates = selector.generate_candidates(100.0)

    assert [candidate.timestamp for candidate in candidates] == pytest.approx(
    [
        10.0,
        25.0,
        40.0,
        55.0,
        70.0,
        85.0,
    ]
)


def test_generated_candidates_are_in_chronological_order() -> None:
    selector = DeterministicFrameSelector()

    candidates = selector.generate_candidates(60.0)

    timestamps = [
        candidate.timestamp
        for candidate in candidates
    ]

    assert timestamps == sorted(timestamps)


def test_selector_avoids_clip_boundaries() -> None:
    selector = DeterministicFrameSelector()

    candidates = selector.generate_candidates(100.0)

    assert candidates[0].timestamp > 0.0
    assert candidates[-1].timestamp < 100.0


def test_duration_must_be_positive() -> None:
    selector = DeterministicFrameSelector()

    with pytest.raises(ValueError, match="duration"):
        selector.generate_candidates(0)

    with pytest.raises(ValueError, match="duration"):
        selector.generate_candidates(-1)


def test_candidate_positions_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="candidate_positions"):
        DeterministicFrameSelector(candidate_positions=())


def test_candidate_positions_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValueError, match="candidate positions"):
        DeterministicFrameSelector(
            candidate_positions=(0.0, 0.5, 0.9)
        )

    with pytest.raises(ValueError, match="candidate positions"):
        DeterministicFrameSelector(
            candidate_positions=(0.1, 1.0)
        )


def test_candidate_positions_must_be_strictly_increasing() -> None:
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        DeterministicFrameSelector(
            candidate_positions=(0.1, 0.4, 0.4, 0.8)
        )


def test_select_returns_middle_candidate() -> None:
    selector = DeterministicFrameSelector()

    candidates = selector.generate_candidates(100.0)

    selected = selector.select(candidates)

    assert selected.timestamp == pytest.approx(55.0)


def test_select_rejects_empty_candidates() -> None:
    selector = DeterministicFrameSelector()

    with pytest.raises(ValueError, match="candidates"):
        selector.select([])