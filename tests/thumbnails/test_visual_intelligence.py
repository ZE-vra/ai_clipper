from pathlib import Path

import numpy as np

from src.thumbnails.perception.local_visual_intelligence import (
    LocalVisualIntelligenceAnalyzer,
)
from src.thumbnails.perception.visual_intelligence import (
    VisualIntelligenceResult,
)


class _Tensor:
    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value

    def tolist(self):
        return self._value.tolist()

    def __len__(self):
        return len(self._value)

    def __getitem__(self, index):
        return _Tensor(self._value[index])


class _Boxes:
    def __init__(self):
        self.conf = _Tensor(np.array([0.9], dtype=float))
        self.xyxy = _Tensor(
            np.array([[20.0, 10.0, 80.0, 90.0]], dtype=float)
        )

    def __len__(self):
        return 1


class _Masks:
    def __init__(self):
        mask = np.zeros((1, 10, 10), dtype=float)
        mask[:, 1:9, 2:8] = 1.0
        self.data = _Tensor(mask)

    def __len__(self):
        return 1


class _Result:
    masks = _Masks()
    boxes = _Boxes()


class _FakeModel:
    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, *_args, **_kwargs):
        return [_Result()]


def test_visual_intelligence_returns_semantic_subject_and_mask(
    tmp_path,
    monkeypatch,
) -> None:
    import src.thumbnails.perception.local_visual_intelligence as module

    monkeypatch.setattr(module, "YOLO", _FakeModel)

    frame = tmp_path / "frame.jpg"
    from PIL import Image

    Image.new("RGB", (100, 100), "white").save(frame)

    result = LocalVisualIntelligenceAnalyzer().analyze(frame)

    assert isinstance(result, VisualIntelligenceResult)
    assert result.analysis_version == "1"
    assert result.primary_subject is not None
    assert result.primary_subject.kind == "person"
    assert result.primary_subject.confidence == 0.9
    assert result.primary_subject.mask_path.is_file()
    assert result.primary_subject.area_ratio > 0.0


def test_visual_intelligence_uses_deterministic_mask_artifact_path(
    tmp_path,
    monkeypatch,
) -> None:
    import src.thumbnails.perception.local_visual_intelligence as module

    monkeypatch.setattr(module, "YOLO", _FakeModel)

    frame = tmp_path / "candidate.jpg"
    from PIL import Image

    Image.new("RGB", (100, 100), "white").save(frame)

    first = LocalVisualIntelligenceAnalyzer().analyze(frame)
    second = LocalVisualIntelligenceAnalyzer().analyze(frame)

    assert first.primary_subject is not None
    assert second.primary_subject is not None
    assert (
        first.primary_subject.mask_path
        == second.primary_subject.mask_path
    )
