from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from driftfm.evaluation import projection

pca = pytest.importorskip("driftfm.evaluation.pca")
ffhq_data = pytest.importorskip("driftfm.data.ffhq_latent")
mnist_data = pytest.importorskip("driftfm.data.mnist_latent")


class _FakeSpine:
    def __init__(self) -> None:
        self.visible = True
        self.color = None
        self.linewidth = None

    def set_visible(self, value: bool) -> None:
        self.visible = value

    def set_color(self, value: str) -> None:
        self.color = value

    def set_linewidth(self, value: float) -> None:
        self.linewidth = value


class _FakeAxis:
    def __init__(self) -> None:
        self.facecolor = None
        self.axisbelow = None
        self.grid_kwargs = None
        self.tick_kwargs = None
        self.x_label = None
        self.y_label = None
        self.spines = {
            "top": _FakeSpine(),
            "right": _FakeSpine(),
            "left": _FakeSpine(),
            "bottom": _FakeSpine(),
        }

    def set_facecolor(self, value: str) -> None:
        self.facecolor = value

    def set_axisbelow(self, value: bool) -> None:
        self.axisbelow = value

    def grid(self, *args, **kwargs) -> None:
        self.grid_kwargs = (args, kwargs)

    def tick_params(self, **kwargs) -> None:
        self.tick_kwargs = kwargs

    def set_xlabel(self, value: str, **kwargs) -> None:
        del kwargs
        self.x_label = value

    def set_ylabel(self, value: str, **kwargs) -> None:
        del kwargs
        self.y_label = value


class _FastUMAP:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.center: np.ndarray | None = None

    def fit(self, points: np.ndarray) -> "_FastUMAP":
        self.center = np.asarray(points, dtype=np.float32).mean(axis=0)
        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        assert self.center is not None
        centered = np.asarray(points, dtype=np.float32) - self.center
        if centered.shape[1] == 1:
            return np.concatenate([centered, np.zeros((centered.shape[0], 1), dtype=np.float32)], axis=1)
        return centered[:, :2]


class _FastTSNE:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit_transform(self, points: np.ndarray) -> np.ndarray:
        centered = np.asarray(points, dtype=np.float32) - np.asarray(points, dtype=np.float32).mean(axis=0)
        if centered.shape[1] == 1:
            return np.concatenate([centered, np.zeros((centered.shape[0], 1), dtype=np.float32)], axis=1)
        return centered[:, :2]


@pytest.fixture
def fast_projection_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    import sklearn.manifold

    monkeypatch.setattr(sklearn.manifold, "TSNE", _FastTSNE)
    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=_FastUMAP))


def test_resolve_plot_semantics_for_ffhq_uses_age_and_gender_encodings() -> None:
    semantics = pca._resolve_plot_semantics(ffhq_data.FFHQ_CLASS_NAMES)

    assert semantics.dataset_kind == "ffhq"
    assert [entry.label for entry in semantics.mixed_legends[0].entries] == [
        "Male children",
        "Male adults",
        "Older males",
        "Female children",
        "Female adults",
        "Older females",
    ]
    assert semantics.class_styles["male_children"].marker == "^"
    assert semantics.class_styles["male_children"].color == "#4E79A7"
    assert semantics.class_styles["female_adult"].marker == "o"
    assert semantics.class_styles["female_adult"].color == "#F28E2B"
    assert semantics.class_styles["female_old"].color == "#E15759"
    assert [legend.title for legend in semantics.mixed_legends] == [None, None]
    assert [entry.label for entry in semantics.mixed_legends[1].entries] == ["Generated", "Real"]
    assert all(len(entry.glyphs) == 2 for entry in semantics.mixed_legends[1].entries)
    assert len(semantics.single_stage_legends) == 1
    assert semantics.single_stage_legends[0] == semantics.mixed_legends[0]


def test_resolve_plot_semantics_for_mnist_uses_digit_labels_and_fill_only_stage_legend() -> None:
    semantics = pca._resolve_plot_semantics(mnist_data.MNIST_CLASS_NAMES)

    assert semantics.dataset_kind == "mnist"
    assert semantics.class_styles["0"].label == "Digit 0"
    assert semantics.class_styles["9"].label == "Digit 9"
    assert semantics.class_styles["0"].marker == "o"
    assert semantics.class_styles["0"].color == "#4E79A7"
    assert [legend.title for legend in semantics.mixed_legends] == ["Digits", None]
    assert semantics.mixed_legends[0].ncol == 2
    assert semantics.mixed_legends[1].title is None
    assert [entry.label for entry in semantics.mixed_legends[1].entries] == ["Generated", "Real"]
    assert all(len(entry.glyphs) == 1 for entry in semantics.mixed_legends[1].entries)
    assert len(semantics.single_stage_legends) == 1


def test_configure_axes_uses_empty_axis_labels() -> None:
    semantics = pca._resolve_plot_semantics(mnist_data.MNIST_CLASS_NAMES)
    axis = _FakeAxis()

    pca._configure_axes(axis, semantics=semantics)

    assert axis.x_label == ""
    assert axis.y_label == ""
    assert axis.spines["top"].visible is False
    assert axis.spines["right"].visible is False


def test_write_pca_artifacts_emits_both_fit_families(tmp_path: Path) -> None:
    class_names = ["alpha", "beta"]
    real_by_class = {
        "alpha": np.asarray([[-2.0, 0.00], [-1.5, 0.05], [-1.0, -0.02]], dtype=np.float32),
        "beta": np.asarray([[1.0, 0.02], [1.5, -0.03], [2.0, 0.00]], dtype=np.float32),
    }
    fake_by_class = {
        "alpha": np.asarray([[0.0, 5.0], [0.2, 5.5], [-0.1, 4.5]], dtype=np.float32),
        "beta": np.asarray([[0.0, -5.0], [0.1, -4.5], [-0.2, -5.5]], dtype=np.float32),
    }

    artifacts = pca.write_pca_artifacts(
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=tmp_path,
    )

    expected_paths = (
        artifacts.plot_path,
        artifacts.real_only_plot_path,
        artifacts.generated_only_plot_path,
        artifacts.real_fit_plot_path,
        artifacts.real_fit_real_only_plot_path,
        artifacts.real_fit_generated_only_plot_path,
        artifacts.projection_path,
        artifacts.real_fit_projection_path,
    )
    for path in expected_paths:
        assert path.exists()

    mixed_fit_projection = json.loads(artifacts.projection_path.read_text(encoding="utf-8"))
    real_fit_projection = json.loads(artifacts.real_fit_projection_path.read_text(encoding="utf-8"))

    assert mixed_fit_projection["fit_basis"] == "fake_real"
    assert real_fit_projection["fit_basis"] == "real_only"
    assert set(mixed_fit_projection["stages"]) == {"fake", "real"}
    assert set(real_fit_projection["stages"]) == {"fake", "real"}

    mixed_fake_alpha = np.asarray(mixed_fit_projection["stages"]["fake"]["alpha"], dtype=np.float32)
    real_fit_fake_alpha = np.asarray(real_fit_projection["stages"]["fake"]["alpha"], dtype=np.float32)
    assert not np.allclose(mixed_fake_alpha, real_fit_fake_alpha)


def test_write_projection_artifacts_emits_method_specific_families(
    tmp_path: Path,
    fast_projection_backends: None,
) -> None:
    class_names = ["alpha", "beta", "gamma"]
    real_by_class = {
        "alpha": np.asarray([[-3.0, 0.0, 0.0], [-2.8, 0.1, 0.0], [-2.6, -0.1, 0.0]], dtype=np.float32),
        "beta": np.asarray([[0.0, 3.0, 0.0], [0.1, 2.8, 0.0], [-0.1, 2.6, 0.0]], dtype=np.float32),
        "gamma": np.asarray([[3.0, 0.0, 0.0], [2.8, 0.1, 0.0], [2.6, -0.1, 0.0]], dtype=np.float32),
    }
    fake_by_class = {
        "alpha": np.asarray([[-2.9, 0.0, 1.0], [-2.7, 0.1, 1.0], [-2.5, -0.1, 1.0]], dtype=np.float32),
        "beta": np.asarray([[0.0, 2.9, 1.0], [0.1, 2.7, 1.0], [-0.1, 2.5, 1.0]], dtype=np.float32),
        "gamma": np.asarray([[2.9, 0.0, 1.0], [2.7, 0.1, 1.0], [2.5, -0.1, 1.0]], dtype=np.float32),
    }

    artifacts = projection.write_projection_artifacts(
        methods=["tsne", "umap", "lda"],
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=tmp_path,
        random_state=123,
    )

    assert set(artifacts) == {"tsne", "umap", "lda"}
    assert artifacts["tsne"].real_fit_plot_path is None
    assert artifacts["tsne"].real_fit_projection_path is None
    assert artifacts["umap"].real_fit_plot_path is not None
    assert artifacts["lda"].real_fit_plot_path is not None

    for method_name, method_artifacts in artifacts.items():
        projection_payload = json.loads(method_artifacts.projection_path.read_text(encoding="utf-8"))
        assert projection_payload["method"] == method_name
        assert projection_payload["fit_basis"] == "fake_real"
        assert set(projection_payload["stages"]) == {"fake", "real"}
        assert method_artifacts.plot_path.exists()
        assert method_artifacts.real_only_plot_path.exists()
        assert method_artifacts.generated_only_plot_path.exists()

    for method_name in ("umap", "lda"):
        real_fit_path = artifacts[method_name].real_fit_projection_path
        assert real_fit_path is not None
        real_fit_payload = json.loads(real_fit_path.read_text(encoding="utf-8"))
        assert real_fit_payload["method"] == method_name
        assert real_fit_payload["fit_basis"] == "real_only"
