# Image generation with Drift Flow Matching

Class-conditional latent generation on MNIST and FFHQ. A single model accepts the state, start time, and interval length, and supports one-step or multi-step sampling. See the [method](docs/methods/drift_flow_matching.md).

## Install

Use Python 3.11. From the repository root:

```bash
cd image_generation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested.txt -e '.[eval]'
```

For GPU execution, install the matching PyTorch CUDA build before the package installation command.

## Generate images

Follow one dataset workflow for data preparation, training, image generation and evaluation at NFE 1, 2 and 5:

| Dataset | Representation | Workflow |
| --- | --- | --- |
| MNIST | 16-dimensional convolutional autoencoder latents | [Train the autoencoder and DFM](docs/datasets/mnist.md) |
| FFHQ | 512-dimensional frozen ALAE latents, six classes | [Import ALAE and train DFM](docs/datasets/ffhq.md) |

Sampling uses EMA weights when available and the same initial seed at each NFE. It saves latent arrays and metadata; `--decoder-checkpoint` also produces image grids.

Both training configurations evaluate at NFE 1, 2, 5, 10, 20 and 50 every 25 epochs and at completion; the standalone evaluation configurations use NFE 1, 2 and 5.

## Paths and devices

YAML paths are relative to the YAML file; command-line paths are relative to the current directory. The supplied `runtime.data_root` and `runtime.runs_root` point to this module's `data/` and `runs/` directories. Edit these roots to use another disk.

Set `trainer.device` and `evaluation.params.device` together. Training batch size must be divisible by `number_of_classes × groups_per_class`.
