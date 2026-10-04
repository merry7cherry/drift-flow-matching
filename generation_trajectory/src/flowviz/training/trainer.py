from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from ..configs import TrainingConfig
from ..data.base import PairDataset
from ..flows.base import FlowMatchingObjective


@dataclass
class TrainingHistory:
    losses: List[float]


def train_model(
    model: torch.nn.Module,
    dataset: PairDataset,
    objective: FlowMatchingObjective,
    config: TrainingConfig,
) -> TrainingHistory:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: List[float] = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for _ in range(config.steps_per_epoch):
            batch = dataset.sample_pairs(config.batch_size, device)
            t = objective.sample_time(config.batch_size, device)
            xt = objective.interpolate(batch.x0, batch.x1, t)
            target_velocity = objective.target_velocity(batch.x0, batch.x1, xt, t)
            predicted_velocity = model(xt, t)
            loss = F.mse_loss(predicted_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(1, config.steps_per_epoch)
        losses.append(epoch_loss)

    return TrainingHistory(losses=losses)


__all__ = ["train_model", "TrainingHistory"]
