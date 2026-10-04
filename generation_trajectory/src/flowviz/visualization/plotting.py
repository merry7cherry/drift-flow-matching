from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_2d_trajectory_figure(trajectories, title, *, target_samples, max_display=128, bounds=None):
    """Plot actual sampler states; reference points have no pathwise correspondence."""
    states = trajectories.detach().cpu().numpy()
    targets = target_samples.detach().cpu().numpy()
    indices = np.linspace(0, states.shape[1] - 1, min(max_display, states.shape[1]), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.scatter(targets[:, 0], targets[:, 1], s=12, c="#94a3b8", alpha=.35, label="Target samples")
    for idx in indices:
        ax.plot(states[:, idx, 0], states[:, idx, 1], c="#64748b", lw=.65, alpha=.3)
    ax.scatter(states[0, indices, 0], states[0, indices, 1], s=13, c="#2563eb", alpha=.6, label="Source")
    ax.scatter(states[-1, :, 0], states[-1, :, 1], s=14, c="#e45b46", alpha=.8, label="Generated")
    if bounds is not None:
        lower, upper = bounds
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def save_figure(figure, path):
    figure.savefig(Path(path), dpi=160)
    plt.close(figure)
