# Image generation with Drift Flow Matching

Class-conditional latent generation on MNIST and FFHQ. A single model accepts the state, start time, and interval length, and supports one-step or multi-step sampling.

Train the class-conditional models from source with the workflows below. Pretrained DFM checkpoints are not distributed; the [reproduction guide](../reproduction/image_generation.md) records the experiment recipes and result provenance.

## Install

Use Python 3.11 or newer; the tested public environment uses Python 3.11.15, PyTorch 2.8.0 and torchvision 0.23.0. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested.txt -e '.[eval,dev]'
```

Install a PyTorch build appropriate for your CUDA environment before this command if needed. The package also supports CPU execution; the small smoke check below validates installation.

The base package provides training, sampling, and preprocessing. `eval` adds OT, FID and optional projection metrics. `data` adds the optional FFHQ downloader. Paths and devices are configured locally.

## First run: a small CPU check

```bash
driftfm-smoke --output-dir ./runs/smoke
```

This generates synthetic six-class, four-dimensional latents, runs two optimizer steps, saves an EMA checkpoint, reloads it, samples at NFE 1 and 2, and evaluates latent OT. It downloads no data or models. `smoke_result.json` records the outputs.

## Train and sample

Prepare the latent data using the [MNIST workflow](docs/datasets/mnist.md) or [FFHQ workflow](docs/datasets/ffhq.md), then run:

```bash
driftfm-train --config configs/experiments/mnist_driftfm.yaml
# or
driftfm-train --config configs/experiments/ffhq_driftfm.yaml
```

Sample any checkpoint produced by this release without loading its training data:

```bash
driftfm-sample \
  --checkpoint runs/mnist_drift/mnist_dfm/checkpoint_final.pt \
  --decoder-checkpoint runs/mnist_ae/latest/ae_final.pt \
  --output-dir runs/mnist_samples --steps 1 2 5 --device cpu
```

Omit `--decoder-checkpoint` to save only latent `.npz` arrays. For FFHQ, pass the imported ALAE checkpoint and a suitable device/batch size. The image grid uses deterministic ALAE decoding; the FFHQ FID evaluation configuration separately enables decoder noise as in the inherited evaluation pipeline.

Each NFE starts from the same seed, with the same model and class labels. Sampling loads EMA weights when present. Output directories contain arrays, optional image grids, and JSON metadata with seed, NFE and checkpoint identity. Only load checkpoints from sources you trust; training checkpoints include optimizer and random-number state.

## Paths and devices

YAML paths are relative to the YAML file. The supplied files under `configs/experiments/` set:

```yaml
runtime:
  data_root: ../../data
  runs_root: ../../runs
```

These resolve to this module's `data/` and `runs/` directories. Set absolute paths or edit the roots to use another disk. `mnist_root` defaults to `{data_root}/mnist`. Other fields can reference `{data_root}`, `{runs_root}`, and `{mnist_root}`. CLI paths are relative to the current directory. Explicit YAML roots take precedence over `DRIFTFM_DATA_ROOT`, `DRIFTFM_RUNS_ROOT`, and `DRIFTFM_MNIST_ROOT`; the environment supplies defaults when a YAML root is omitted.

Set `trainer.device` and `evaluation.params.device` together. Training batch size must be divisible by `number_of_classes × groups_per_class`. The full configurations use large batches (MNIST 16,000; FFHQ 3,072), so they are not laptop quickstarts. Reducing a batch or training budget changes the experiment.

A run saves its resolved absolute paths, architecture, optimizer, EMA, RNG state and seed. `latest` in a path resolves through the local `LATEST_RUN` marker; it is optional convenience, not a download. For durable comparisons, record the concrete checkpoint path.

## Evaluation and resume

```bash
driftfm-evaluate \
  --checkpoint runs/mnist_drift/mnist_dfm/checkpoint_final.pt \
  --config configs/experiments/mnist_eval.yaml

driftfm-train --config configs/experiments/mnist_driftfm.yaml \
  --resume-from runs/mnist_drift/mnist_dfm/checkpoint_epoch25.pt
```

Evaluation writes one directory per NFE and an `evaluation_index.json`. Full experiment configuration overrides relocate both data and evaluation paths; an evaluation-only configuration changes evaluation settings. Use the dataset-specific CLIs (`driftfm-eval-mnist`, `driftfm-eval-ffhq`) for explicit decoder, test-data and metric options.

Keep the dataset, model and optimization settings unchanged when resuming. Increase the configured epoch budget if continuing a completed run. Starting a new experiment must use a new `output.run_name` if the target run already contains checkpoints. Accidental overwrites are rejected. The smoke command likewise expects a fresh output directory.

## Code map and verification

- `src/driftfm/methods/drift_flow_matching.py`: grouped Sinkhorn drift, training objective and interval sampling.
- `src/driftfm/architectures/`: conditional MLP and image encoders/decoders.
- `src/driftfm/data/`: balanced class latent datasets.
- `src/driftfm/training/`: optimizer loop, checkpoint recovery and EMA.
- `src/driftfm/evaluation/`: latent/image OT, classifier accuracy, ALAE-decoded FID and paper PCA/UMAP visualizations.
- `src/driftfm/tasks/`: command-line entrypoints.

```bash
python -m pip install -e '.[eval,dev,data]'
python -m pytest -q
```

Tests cover drift math, labels, preprocessing, checkpoint resume/RNG state, evaluation, configurable paths, and synthetic train/sample/evaluate. See the [verification record](../reproduction/VERIFICATION.md) for tested environments and validation scope.

See [method details](docs/methods/drift_flow_matching.md), [architecture](docs/architecture.md), and the repository's [reproduction guide](../reproduction/image_generation.md). ALAE-derived code retains its upstream copyright and Apache 2.0 notice; see the root [third-party notices](../THIRD_PARTY_NOTICES.md).
