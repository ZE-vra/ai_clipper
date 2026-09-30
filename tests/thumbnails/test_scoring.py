import pytest

from src.thumbnails.perception.scoring import (
    FrameCandidateScore,
    FrameCandidateScoringConfig,
)


def test_frame_candidate_score_stores_scores() -> None:
    score = FrameCandidateScore(
        quality_score=0.8,
        subject_score=0.7,
        focal_score=0.6,
        overall_score=0.72,
    )

    assert score.quality_score == pytest.approx(0.8)
    assert score.subject_score == pytest.approx(0.7)
    assert score.focal_score == pytest.approx(0.6)
    assert score.overall_score == pytest.approx(0.72)


@pytest.mark.parametrize(
    "field",
    (
        "quality_score",
        "subject_score",
        "focal_score",
        "overall_score",
    ),
)
def test_frame_candidate_score_rejects_invalid_scores(
    field: str,
) -> None:
    values = {
        "quality_score": 0.5,
        "subject_score": 0.5,
        "focal_score": 0.5,
        "overall_score": 0.5,
    }

    values[field] = -0.1

    with pytest.raises(ValueError):
        FrameCandidateScore(**values)


def test_scoring_config_has_expected_defaults() -> None:
    config = FrameCandidateScoringConfig()

    assert config.quality_weight == pytest.approx(0.40)
    assert config.subject_weight == pytest.approx(0.35)
    assert config.focal_weight == pytest.approx(0.25)


def test_scoring_config_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError):
        FrameCandidateScoringConfig(
            quality_weight=0.0,
            subject_weight=0.0,
            focal_weight=0.0,
        )


def test_scoring_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        FrameCandidateScoringConfig(
            quality_weight=-0.1,
        )