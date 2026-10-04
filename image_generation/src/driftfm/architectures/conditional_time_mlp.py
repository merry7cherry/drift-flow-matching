from __future__ import annotations

import torch
import torch.nn as nn

from ..data import Conditioning
from .base import VelocityModel, resolve_conditioning_mode


def _build_activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU(inplace=True)
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _resolve_class_indices(
    conditioning: Conditioning | None,
    *,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    if conditioning is None or conditioning.class_labels is None:
        raise ValueError("conditioning.class_labels is required for class-conditional generation")

    labels = conditioning.class_labels.to(device=device, dtype=torch.long).reshape(-1)
    if labels.shape[0] != batch_size:
        raise ValueError(f"class_labels must have shape [{batch_size}], got {tuple(labels.shape)}")

    min_label = int(labels.min().item())
    max_label = int(labels.max().item())
    if min_label < 0:
        raise ValueError(f"class_labels must be >= 0, got min={min_label}")
    if max_label >= num_classes:
        raise ValueError(f"class_labels must be smaller than {num_classes}, got max={max_label}")
    return labels


class ConditionalTimeMLP(VelocityModel):
    def __init__(
        self,
        *,
        data_dim: int | None = None,
        input_dim: int | None = None,
        hidden_sizes: list[int] | tuple[int, ...],
        num_classes: int | None = None,
        conditioning_mode: str | None = None,
        class_embedding_dim: int | None = None,
        class_emb_dim: int | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if data_dim is None:
            data_dim = input_dim
        if data_dim is None:
            raise ValueError("data_dim or input_dim must be provided")
        if class_embedding_dim is None:
            class_embedding_dim = 32 if class_emb_dim is None else class_emb_dim

        self.data_dim = data_dim
        self.num_classes = num_classes
        self.conditioning_mode = resolve_conditioning_mode(conditioning_mode, num_classes=num_classes)
        use_class_condition = self.conditioning_mode in {"class"}
        self.class_embedding_dim = class_embedding_dim if use_class_condition else 0
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim) if use_class_condition else None
        input_dim = data_dim + 2 + self.class_embedding_dim
        layer_sizes = [input_dim, *hidden_sizes, data_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != layer_sizes[-1]:
                layers.append(_build_activation(activation))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        conditioning: Conditioning | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if conditioning is None and class_labels is not None:
            conditioning = Conditioning(class_labels=class_labels)

        original_shape = x_t.shape
        x_flat = x_t.reshape(x_t.shape[0], -1)
        t_flat = t.reshape(t.shape[0], -1)
        h_flat = h.reshape(h.shape[0], -1)
        if t_flat.shape[0] != x_flat.shape[0] or h_flat.shape[0] != x_flat.shape[0]:
            raise ValueError(
                "x_t, t, and h must share the same batch axis, "
                f"got x_t={tuple(x_t.shape)}, t={tuple(t.shape)}, h={tuple(h.shape)}"
            )
        pieces = [x_flat, t_flat, h_flat]

        if self.class_embedding is not None:
            assert self.num_classes is not None
            indices = _resolve_class_indices(
                conditioning,
                batch_size=x_flat.shape[0],
                device=x_t.device,
                num_classes=self.num_classes,
            )
            class_features = self.class_embedding(indices).to(dtype=x_flat.dtype)
            pieces.append(class_features)
        output = self.network(torch.cat(pieces, dim=-1))
        return output.reshape(original_shape)
