from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PLOT_STAGE_ORDER = ("fake", "real")
PROJECTION_METHODS = ("pca", "tsne", "umap", "lda")
_CLASS_PALETTE = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
_FFHQ_AGE_COLORS = {
    "children": "#4E79A7",
    "adult": "#F28E2B",
    "old": "#E15759",
}
_FFHQ_GENDER_MARKERS = {
    "male": "^",
    "female": "o",
}
_FFHQ_CLASS_NAMES = (
    "male_children",
    "male_adult",
    "male_old",
    "female_children",
    "female_adult",
    "female_old",
)
_FFHQ_CLASS_LABELS = {
    "male_children": "Male children",
    "male_adult": "Male adults",
    "male_old": "Older males",
    "female_children": "Female children",
    "female_adult": "Female adults",
    "female_old": "Older females",
}
_SAMPLE_LEGEND_COLOR = "#6B7280"
_SAMPLE_LEGEND_EDGE = "#4B5563"
_FAKE_REAL_FIT_BASIS = "fake_real"
_REAL_ONLY_FIT_BASIS = "real_only"


@dataclass(slots=True)
class ProjectionArtifacts:
    plot_path: Path
    real_only_plot_path: Path
    generated_only_plot_path: Path
    projection_path: Path
    real_fit_plot_path: Path | None = None
    real_fit_real_only_plot_path: Path | None = None
    real_fit_generated_only_plot_path: Path | None = None
    real_fit_projection_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ProjectionArtifactFamilySpec:
    method_name: str
    fit_basis: str
    fit_stage_names: tuple[str, ...]
    plot_filename: str
    real_only_plot_filename: str
    generated_only_plot_filename: str
    projection_filename: str


@dataclass(frozen=True, slots=True)
class ProjectionArtifactFamily:
    plot_path: Path
    real_only_plot_path: Path
    generated_only_plot_path: Path
    projection_path: Path


@dataclass(frozen=True, slots=True)
class ProjectionClassStyle:
    label: str
    color: str
    marker: str
    generated_size: float
    real_size: float


@dataclass(frozen=True, slots=True)
class ProjectionMarkerGlyph:
    color: str
    marker: str
    filled: bool = True
    markersize: float = 7.0


@dataclass(frozen=True, slots=True)
class ProjectionLegendEntry:
    label: str
    glyphs: tuple[ProjectionMarkerGlyph, ...]


@dataclass(frozen=True, slots=True)
class ProjectionLegendSpec:
    title: str | None
    entries: tuple[ProjectionLegendEntry, ...]
    loc: str
    anchor: tuple[float, float]
    ncol: int = 1


@dataclass(frozen=True, slots=True)
class ProjectionPlotSemantics:
    dataset_kind: str
    class_styles: dict[str, ProjectionClassStyle]
    mixed_legends: tuple[ProjectionLegendSpec, ...]
    single_stage_legends: tuple[ProjectionLegendSpec, ...]
    x_label: str = ""
    y_label: str = ""


def _filenames(method_name: str, *, real_fit: bool = False) -> tuple[str, str, str, str]:
    prefix = f"latent_{method_name}"
    if real_fit:
        prefix = f"{prefix}_real_fit"
    return (
        f"{prefix}.png",
        f"{prefix}_real_only.png",
        f"{prefix}_generated_only.png",
        f"{prefix}_projection.json",
    )


def _family_spec(method_name: str, *, fit_basis: str) -> ProjectionArtifactFamilySpec:
    real_fit = fit_basis == _REAL_ONLY_FIT_BASIS
    plot_filename, real_only_filename, generated_only_filename, projection_filename = _filenames(
        method_name,
        real_fit=real_fit,
    )
    return ProjectionArtifactFamilySpec(
        method_name=method_name,
        fit_basis=fit_basis,
        fit_stage_names=("real",) if real_fit else PLOT_STAGE_ORDER,
        plot_filename=plot_filename,
        real_only_plot_filename=real_only_filename,
        generated_only_plot_filename=generated_only_filename,
        projection_filename=projection_filename,
    )


def _project_bounds(projected_by_stage: dict[str, dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    all_points = [
        points
        for by_class in projected_by_stage.values()
        for points in by_class.values()
        if points.size > 0
    ]
    stacked = np.concatenate(all_points, axis=0)
    min_x, min_y = stacked.min(axis=0)
    max_x, max_y = stacked.max(axis=0)
    pad_x = max(1e-6, 0.05 * (max_x - min_x if max_x > min_x else 1.0))
    pad_y = max(1e-6, 0.05 * (max_y - min_y if max_y > min_y else 1.0))
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _ensure_two_columns(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError("Projected points must be a 2D array")
    if points.shape[1] == 2:
        return points.astype(np.float32, copy=False)
    if points.shape[1] == 1:
        return np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
    return points[:, :2].astype(np.float32, copy=False)


def _project_stage(
    model: object,
    by_class: dict[str, np.ndarray],
    *,
    class_names: list[str],
) -> dict[str, np.ndarray]:
    return {
        class_name: _ensure_two_columns(model.transform(np.asarray(by_class[class_name], dtype=np.float32)))
        for class_name in class_names
    }


def _ffhq_class_style(class_name: str) -> ProjectionClassStyle:
    gender, age_group = class_name.split("_", 1)
    return ProjectionClassStyle(
        label=_FFHQ_CLASS_LABELS[class_name],
        color=_FFHQ_AGE_COLORS[age_group],
        marker=_FFHQ_GENDER_MARKERS[gender],
        generated_size=42.0 if gender == "male" else 34.0,
        real_size=64.0 if gender == "male" else 54.0,
    )


def _mnist_class_style(class_name: str, *, class_index: int) -> ProjectionClassStyle:
    return ProjectionClassStyle(
        label=f"Digit {class_name}",
        color=_CLASS_PALETTE[class_index % len(_CLASS_PALETTE)],
        marker="o",
        generated_size=34.0,
        real_size=54.0,
    )


def _generic_class_style(class_name: str, *, class_index: int) -> ProjectionClassStyle:
    return ProjectionClassStyle(
        label=class_name,
        color=_CLASS_PALETTE[class_index % len(_CLASS_PALETTE)],
        marker="o",
        generated_size=34.0,
        real_size=54.0,
    )


def _build_ffhq_semantics(class_names: list[str]) -> ProjectionPlotSemantics:
    class_styles = {class_name: _ffhq_class_style(class_name) for class_name in class_names}
    class_legend = ProjectionLegendSpec(
        title=None,
        entries=tuple(
            ProjectionLegendEntry(
                label=class_styles[class_name].label,
                glyphs=(
                    ProjectionMarkerGlyph(
                        color=class_styles[class_name].color,
                        marker=class_styles[class_name].marker,
                        filled=True,
                    ),
                ),
            )
            for class_name in class_names
        ),
        loc="upper right",
        anchor=(0.985, 0.985),
    )
    stage_legend = ProjectionLegendSpec(
        title=None,
        entries=(
            ProjectionLegendEntry(
                label="Generated",
                glyphs=(
                    ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_COLOR, marker="o", filled=True),
                    ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_COLOR, marker="^", filled=True),
                ),
            ),
            ProjectionLegendEntry(
                label="Real",
                glyphs=(
                    ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_EDGE, marker="o", filled=False, markersize=8.0),
                    ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_EDGE, marker="^", filled=False, markersize=8.0),
                ),
            ),
        ),
        loc="lower right",
        anchor=(0.985, 0.02),
    )
    return ProjectionPlotSemantics(
        dataset_kind="ffhq",
        class_styles=class_styles,
        mixed_legends=(class_legend, stage_legend),
        single_stage_legends=(class_legend,),
    )


def _build_mnist_semantics(class_names: list[str]) -> ProjectionPlotSemantics:
    class_styles = {
        class_name: _mnist_class_style(class_name, class_index=class_index)
        for class_index, class_name in enumerate(class_names)
    }
    class_legend = ProjectionLegendSpec(
        title="Digits",
        entries=tuple(
            ProjectionLegendEntry(
                label=class_styles[class_name].label,
                glyphs=(
                    ProjectionMarkerGlyph(
                        color=class_styles[class_name].color,
                        marker="o",
                        filled=True,
                    ),
                ),
            )
            for class_name in class_names
        ),
        loc="upper right",
        anchor=(0.985, 0.985),
        ncol=2,
    )
    stage_legend = ProjectionLegendSpec(
        title=None,
        entries=(
            ProjectionLegendEntry(
                label="Generated",
                glyphs=(ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_COLOR, marker="o", filled=True),),
            ),
            ProjectionLegendEntry(
                label="Real",
                glyphs=(ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_EDGE, marker="o", filled=False, markersize=8.0),),
            ),
        ),
        loc="lower right",
        anchor=(0.985, 0.02),
    )
    return ProjectionPlotSemantics(
        dataset_kind="mnist",
        class_styles=class_styles,
        mixed_legends=(class_legend, stage_legend),
        single_stage_legends=(class_legend,),
    )


def _build_generic_semantics(class_names: list[str]) -> ProjectionPlotSemantics:
    class_styles = {
        class_name: _generic_class_style(class_name, class_index=class_index)
        for class_index, class_name in enumerate(class_names)
    }
    class_legend = ProjectionLegendSpec(
        title="Classes",
        entries=tuple(
            ProjectionLegendEntry(
                label=class_styles[class_name].label,
                glyphs=(
                    ProjectionMarkerGlyph(
                        color=class_styles[class_name].color,
                        marker="o",
                        filled=True,
                    ),
                ),
            )
            for class_name in class_names
        ),
        loc="upper right",
        anchor=(0.985, 0.985),
        ncol=2 if len(class_names) > 6 else 1,
    )
    stage_legend = ProjectionLegendSpec(
        title=None,
        entries=(
            ProjectionLegendEntry(
                label="Generated",
                glyphs=(ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_COLOR, marker="o", filled=True),),
            ),
            ProjectionLegendEntry(
                label="Real",
                glyphs=(ProjectionMarkerGlyph(color=_SAMPLE_LEGEND_EDGE, marker="o", filled=False, markersize=8.0),),
            ),
        ),
        loc="lower right",
        anchor=(0.985, 0.02),
    )
    return ProjectionPlotSemantics(
        dataset_kind="generic",
        class_styles=class_styles,
        mixed_legends=(class_legend, stage_legend),
        single_stage_legends=(class_legend,),
    )


def _resolve_plot_semantics(class_names: list[str]) -> ProjectionPlotSemantics:
    if class_names and all(class_name in _FFHQ_CLASS_NAMES for class_name in class_names):
        return _build_ffhq_semantics(class_names)
    if class_names and all(class_name.isdigit() for class_name in class_names):
        return _build_mnist_semantics(class_names)
    return _build_generic_semantics(class_names)


def _configure_axes(ax: object, *, semantics: ProjectionPlotSemantics) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color="#D9DCE1", linewidth=0.8, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#A8AEB8")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", colors="#48505E", labelsize=11)
    ax.set_xlabel(semantics.x_label, fontsize=12, color="#2F3542")
    ax.set_ylabel(semantics.y_label, fontsize=12, color="#2F3542")


def _legend_handle(entry: ProjectionLegendEntry, *, line2d_type: type[object]) -> object:
    glyph_handles = tuple(
        line2d_type(
            [0],
            [0],
            marker=glyph.marker,
            linestyle="",
            markerfacecolor=(glyph.color if glyph.filled else "white"),
            markeredgecolor=glyph.color,
            markeredgewidth=1.25 if not glyph.filled else 1.0,
            markersize=glyph.markersize,
        )
        for glyph in entry.glyphs
    )
    return glyph_handles[0] if len(glyph_handles) == 1 else glyph_handles


def _legend_specs_for_stages(
    semantics: ProjectionPlotSemantics,
    *,
    visible_stages: tuple[str, ...],
) -> tuple[ProjectionLegendSpec, ...]:
    return semantics.mixed_legends if tuple(visible_stages) == PLOT_STAGE_ORDER else semantics.single_stage_legends


def _style_legend(legend: object) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.92)
    frame.set_edgecolor("#D9DCE1")
    frame.set_linewidth(0.9)


def _plot_stage_points(
    ax: object,
    *,
    class_style: ProjectionClassStyle,
    points: np.ndarray,
    stage_name: str,
) -> None:
    if stage_name == "fake":
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=class_style.generated_size,
            marker=class_style.marker,
            color=class_style.color,
            alpha=0.42,
            linewidths=0.0,
            zorder=2,
        )
        return
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=class_style.real_size,
        marker=class_style.marker,
        facecolors="none",
        edgecolors=class_style.color,
        linewidths=1.25,
        alpha=0.95,
        zorder=3,
    )


def _write_plot(
    *,
    projected_by_stage: dict[str, dict[str, np.ndarray]],
    class_names: list[str],
    bounds: tuple[float, float, float, float],
    visible_stages: tuple[str, ...],
    plot_path: Path,
) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("matplotlib is required for latent projection evaluation artifacts.") from exc

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    semantics = _resolve_plot_semantics(class_names)
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    _configure_axes(ax, semantics=semantics)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    for class_name in class_names:
        style = semantics.class_styles[class_name]
        for stage_name in visible_stages:
            _plot_stage_points(
                ax,
                class_style=style,
                points=projected_by_stage[stage_name][class_name],
                stage_name=stage_name,
            )

    legend_specs = _legend_specs_for_stages(semantics, visible_stages=visible_stages)
    handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.8)}
    for legend_index, legend_spec in enumerate(legend_specs):
        handles = [_legend_handle(entry, line2d_type=Line2D) for entry in legend_spec.entries]
        legend_kwargs = {
            "handles": handles,
            "labels": [entry.label for entry in legend_spec.entries],
            "loc": legend_spec.loc,
            "bbox_to_anchor": legend_spec.anchor,
            "borderaxespad": 0.0,
            "frameon": True,
            "fontsize": 10,
            "ncol": legend_spec.ncol,
            "handler_map": handler_map,
        }
        if legend_spec.title is not None:
            legend_kwargs["title"] = legend_spec.title
            legend_kwargs["title_fontsize"] = 10.5
        legend = ax.legend(**legend_kwargs)
        _style_legend(legend)
        if legend_index < len(legend_specs) - 1:
            ax.add_artist(legend)

    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stack_stage_points(
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    stage_names: tuple[str, ...],
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    pieces: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for stage_name in stage_names:
        for class_index, class_name in enumerate(class_names):
            points = np.asarray(by_stage[stage_name][class_name], dtype=np.float32)
            pieces.append(points)
            labels.append(np.full((points.shape[0],), class_index, dtype=np.int64))
    if not pieces:
        raise ValueError("Latent projection requires at least one class")
    return np.concatenate(pieces, axis=0), np.concatenate(labels, axis=0)


def _split_projected_points(
    projected: np.ndarray,
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    stage_names: tuple[str, ...],
    class_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    offset = 0
    projected_by_stage = {stage_name: {} for stage_name in PLOT_STAGE_ORDER}
    for stage_name in stage_names:
        for class_name in class_names:
            count = int(by_stage[stage_name][class_name].shape[0])
            projected_by_stage[stage_name][class_name] = _ensure_two_columns(projected[offset : offset + count])
            offset += count
    return projected_by_stage


def _project_by_stage(
    model: object,
    *,
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "fake": _project_stage(model, fake_by_class, class_names=class_names),
        "real": _project_stage(model, real_by_class, class_names=class_names),
    }


def _safe_tsne_perplexity(n_samples: int) -> float:
    if n_samples < 2:
        raise ValueError("t-SNE projection requires at least 2 latent points")
    return float(min(30.0, max(1.0, (n_samples - 1) / 3.0)))


def _safe_umap_neighbors(n_samples: int) -> int:
    if n_samples < 3:
        raise ValueError("UMAP projection requires at least 3 latent points")
    return min(15, n_samples - 1)


def _project_with_pca(
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    class_names: list[str],
    family_spec: ProjectionArtifactFamilySpec,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], np.ndarray | None]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("scikit-learn is required for PCA evaluation artifacts.") from exc

    fitted_points, _ = _stack_stage_points(by_stage, stage_names=family_spec.fit_stage_names, class_names=class_names)
    model = PCA(n_components=2)
    model.fit(fitted_points)
    return (
        _project_by_stage(model, fake_by_class=by_stage["fake"], real_by_class=by_stage["real"], class_names=class_names),
        {"n_components": 2},
        model.explained_variance_ratio_,
    )


def _project_with_tsne(
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    class_names: list[str],
    random_state: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], np.ndarray | None]:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("scikit-learn is required for t-SNE evaluation artifacts.") from exc

    fitted_points, _ = _stack_stage_points(by_stage, stage_names=PLOT_STAGE_ORDER, class_names=class_names)
    perplexity = _safe_tsne_perplexity(int(fitted_points.shape[0]))
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        max_iter=250,
        random_state=int(random_state),
    )
    projected = model.fit_transform(fitted_points)
    return (
        _split_projected_points(projected, by_stage, stage_names=PLOT_STAGE_ORDER, class_names=class_names),
        {"n_components": 2, "perplexity": perplexity, "random_state": int(random_state)},
        None,
    )


def _project_with_umap(
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    class_names: list[str],
    family_spec: ProjectionArtifactFamilySpec,
    random_state: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], np.ndarray | None]:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP evaluation artifacts. "
            "Install the evaluation extras or run `pip install umap-learn`."
        ) from exc

    fitted_points, _ = _stack_stage_points(by_stage, stage_names=family_spec.fit_stage_names, class_names=class_names)
    n_neighbors = _safe_umap_neighbors(int(fitted_points.shape[0]))
    model = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        init="random",
        n_epochs=50,
        random_state=int(random_state),
    )
    model.fit(fitted_points)
    return (
        _project_by_stage(model, fake_by_class=by_stage["fake"], real_by_class=by_stage["real"], class_names=class_names),
        {"n_components": 2, "n_neighbors": n_neighbors, "random_state": int(random_state)},
        None,
    )


def _project_with_lda(
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    class_names: list[str],
    family_spec: ProjectionArtifactFamilySpec,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], np.ndarray | None]:
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError as exc:
        raise ImportError("scikit-learn is required for LDA evaluation artifacts.") from exc

    fitted_points, class_labels = _stack_stage_points(
        by_stage,
        stage_names=family_spec.fit_stage_names,
        class_names=class_names,
    )
    n_components = min(2, len(class_names) - 1, int(fitted_points.shape[1]))
    if n_components < 1:
        raise ValueError("LDA projection requires at least 2 classes and 1 latent feature")
    model = LinearDiscriminantAnalysis(n_components=n_components)
    model.fit(fitted_points, class_labels)
    return (
        _project_by_stage(model, fake_by_class=by_stage["fake"], real_by_class=by_stage["real"], class_names=class_names),
        {"n_components": int(n_components), "label_source": "class"},
        None,
    )


def _compute_projection(
    by_stage: dict[str, dict[str, np.ndarray]],
    *,
    class_names: list[str],
    family_spec: ProjectionArtifactFamilySpec,
    random_state: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], np.ndarray | None]:
    if family_spec.method_name == "pca":
        return _project_with_pca(by_stage, class_names=class_names, family_spec=family_spec)
    if family_spec.method_name == "tsne":
        return _project_with_tsne(by_stage, class_names=class_names, random_state=random_state)
    if family_spec.method_name == "umap":
        return _project_with_umap(
            by_stage,
            class_names=class_names,
            family_spec=family_spec,
            random_state=random_state,
        )
    if family_spec.method_name == "lda":
        return _project_with_lda(by_stage, class_names=class_names, family_spec=family_spec)
    raise ValueError(f"Unsupported latent projection method: {family_spec.method_name}")


def _write_projection_json(
    *,
    output_root: Path,
    projected_by_stage: dict[str, dict[str, np.ndarray]],
    class_names: list[str],
    family_spec: ProjectionArtifactFamilySpec,
    metadata: dict[str, Any],
    explained_variance_ratio: np.ndarray | None,
) -> Path:
    projection_path = output_root / family_spec.projection_filename
    payload: dict[str, Any] = {
        "method": family_spec.method_name,
        "fit_basis": family_spec.fit_basis,
        "class_names": list(class_names),
        "projector": metadata,
        "stages": {
            stage_name: {
                class_name: projected_by_stage[stage_name][class_name].astype(float).tolist()
                for class_name in class_names
            }
            for stage_name in PLOT_STAGE_ORDER
        },
    }
    if explained_variance_ratio is not None:
        payload["explained_variance_ratio"] = explained_variance_ratio.astype(float).tolist()
    with projection_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return projection_path


def _write_artifact_family(
    *,
    output_root: Path,
    by_stage: dict[str, dict[str, np.ndarray]],
    class_names: list[str],
    family_spec: ProjectionArtifactFamilySpec,
    random_state: int,
) -> ProjectionArtifactFamily:
    projected_by_stage, metadata, explained_variance_ratio = _compute_projection(
        by_stage,
        class_names=class_names,
        family_spec=family_spec,
        random_state=random_state,
    )
    projection_path = _write_projection_json(
        output_root=output_root,
        projected_by_stage=projected_by_stage,
        class_names=class_names,
        family_spec=family_spec,
        metadata=metadata,
        explained_variance_ratio=explained_variance_ratio,
    )

    bounds = _project_bounds(projected_by_stage)
    plot_path = output_root / family_spec.plot_filename
    real_only_plot_path = output_root / family_spec.real_only_plot_filename
    generated_only_plot_path = output_root / family_spec.generated_only_plot_filename

    _write_plot(
        projected_by_stage=projected_by_stage,
        class_names=class_names,
        bounds=bounds,
        visible_stages=PLOT_STAGE_ORDER,
        plot_path=plot_path,
    )
    _write_plot(
        projected_by_stage=projected_by_stage,
        class_names=class_names,
        bounds=bounds,
        visible_stages=("real",),
        plot_path=real_only_plot_path,
    )
    _write_plot(
        projected_by_stage=projected_by_stage,
        class_names=class_names,
        bounds=bounds,
        visible_stages=("fake",),
        plot_path=generated_only_plot_path,
    )
    return ProjectionArtifactFamily(
        plot_path=plot_path.resolve(),
        real_only_plot_path=real_only_plot_path.resolve(),
        generated_only_plot_path=generated_only_plot_path.resolve(),
        projection_path=projection_path.resolve(),
    )


def write_method_artifacts(
    *,
    method_name: str,
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str | Path,
    random_state: int,
) -> ProjectionArtifacts:
    if method_name not in PROJECTION_METHODS:
        raise ValueError(f"Unsupported latent projection method: {method_name}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    by_stage = {
        "fake": fake_by_class,
        "real": real_by_class,
    }
    fake_real_family = _write_artifact_family(
        output_root=output_root,
        by_stage=by_stage,
        class_names=class_names,
        family_spec=_family_spec(method_name, fit_basis=_FAKE_REAL_FIT_BASIS),
        random_state=random_state,
    )
    if method_name == "tsne":
        return ProjectionArtifacts(
            plot_path=fake_real_family.plot_path,
            real_only_plot_path=fake_real_family.real_only_plot_path,
            generated_only_plot_path=fake_real_family.generated_only_plot_path,
            projection_path=fake_real_family.projection_path,
        )

    real_fit_family = _write_artifact_family(
        output_root=output_root,
        by_stage=by_stage,
        class_names=class_names,
        family_spec=_family_spec(method_name, fit_basis=_REAL_ONLY_FIT_BASIS),
        random_state=random_state,
    )
    return ProjectionArtifacts(
        plot_path=fake_real_family.plot_path,
        real_only_plot_path=fake_real_family.real_only_plot_path,
        generated_only_plot_path=fake_real_family.generated_only_plot_path,
        projection_path=fake_real_family.projection_path,
        real_fit_plot_path=real_fit_family.plot_path,
        real_fit_real_only_plot_path=real_fit_family.real_only_plot_path,
        real_fit_generated_only_plot_path=real_fit_family.generated_only_plot_path,
        real_fit_projection_path=real_fit_family.projection_path,
    )


def write_projection_artifacts(
    *,
    methods: list[str],
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str | Path,
    random_state: int,
) -> dict[str, ProjectionArtifacts]:
    return {
        method_name: write_method_artifacts(
            method_name=method_name,
            fake_by_class=fake_by_class,
            real_by_class=real_by_class,
            class_names=class_names,
            output_dir=output_dir,
            random_state=random_state,
        )
        for method_name in methods
    }


def write_pca_artifacts(
    *,
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str | Path,
    random_state: int = 42,
) -> ProjectionArtifacts:
    return write_method_artifacts(
        method_name="pca",
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=output_dir,
        random_state=random_state,
    )


def write_tsne_artifacts(
    *,
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str | Path,
    random_state: int,
) -> ProjectionArtifacts:
    return write_method_artifacts(
        method_name="tsne",
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=output_dir,
        random_state=random_state,
    )


def write_umap_artifacts(
    *,
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str | Path,
    random_state: int,
) -> ProjectionArtifacts:
    return write_method_artifacts(
        method_name="umap",
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=output_dir,
        random_state=random_state,
    )


def write_lda_artifacts(
    *,
    fake_by_class: dict[str, np.ndarray],
    real_by_class: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str | Path,
    random_state: int = 42,
) -> ProjectionArtifacts:
    del random_state
    return write_method_artifacts(
        method_name="lda",
        fake_by_class=fake_by_class,
        real_by_class=real_by_class,
        class_names=class_names,
        output_dir=output_dir,
        random_state=42,
    )
