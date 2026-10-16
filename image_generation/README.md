# Image generation with Drift Flow Matching

Class-conditional latent generation on MNIST and FFHQ. A single model accepts the state, start time, and interval length, and supports one-step or multi-step sampling.

## Install

Use Python 3.11. From the repository root:

```bash
cd image_generation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested.txt -e '.[eval]'
```

For GPU execution, install the matching PyTorch CUDA build before the package installation command. CPU execution is also supported.

The `eval` extra adds OT, FID and PCA/UMAP evaluation. The optional `data` extra adds the FFHQ downloader.

## First run: a small CPU check

```bash
driftfm-smoke --output-dir ./runs/smoke
```

This checks training, checkpoint loading, sampling and latent OT on small synthetic data. Results are saved in `runs/smoke/smoke_result.json`. Use a fresh output directory for each check.

## Generate images

Follow one dataset workflow for data preparation, training, image generation and evaluation at NFE 1, 2 and 5:

| Dataset | Representation | Workflow |
| --- | --- | --- |
| MNIST | 16-dimensional convolutional autoencoder latents | [Train the autoencoder and DFM](docs/datasets/mnist.md) |
| FFHQ | 512-dimensional frozen ALAE latents, six classes | [Import ALAE and train DFM](docs/datasets/ffhq.md) |

Sampling uses EMA weights when available and the same initial seed at each NFE. It saves latent arrays, image grids and metadata. Omit `--decoder-checkpoint` to save only latents.

## Paths and devices

YAML paths are relative to the YAML file. The supplied files under `configs/experiments/` set:

```yaml
runtime:
  data_root: ../../data
  runs_root: ../../runs
```

These resolve to this module's `data/` and `runs/` directories. Edit the roots to use another disk. CLI paths are relative to the current directory. A `latest` path resolves to the most recent local run through its `LATEST_RUN` marker.

Set `trainer.device` and `evaluation.params.device` together. Training batch size must be divisible by `number_of_classes × groups_per_class`. The supplied training budgets are listed in each dataset guide.

## Resume training

```bash
driftfm-train --config configs/experiments/mnist_driftfm.yaml \
  --resume-from runs/mnist_drift/mnist_dfm/checkpoint_epoch25.pt
```

Keep the dataset, model and optimization settings unchanged when resuming; increase the epoch budget to continue a completed run. Use a new `output.run_name` for a new experiment.

## Development

See the [method](docs/methods/drift_flow_matching.md) and [code map](docs/architecture.md).

```bash
python -m pip install -c constraints-tested.txt -e '.[eval,dev,data]'
python -m pytest -q
```

See the [verification record](../reproduction/VERIFICATION.md) for tested environments and checks.
