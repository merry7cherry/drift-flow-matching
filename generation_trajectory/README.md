# Generation Trajectory

Train Drift Flow Matching (DFM) on four synthetic 2D targets: checkerboard,
letters F and M, and two moons. The module includes Flow Matching and MeanFlow
baselines. Data are generated in memory.

## Install

Use Python 3.11. From the repository root:

```bash
cd generation_trajectory
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Train and view trajectories

Run the two-moons demo on CPU with 1,000 optimizer updates:

```bash
dfm-trajectories --output outputs/demo
```

Open the NFE 1 and 20 figures to compare the learned sampling paths. The command
also saves the trained checkpoint:

```text
outputs/demo/
  2d_circular_uniform_to_moon_drift_flow_matching_nfe_1.png
  2d_circular_uniform_to_moon_drift_flow_matching_nfe_20.png
  checkpoints/2d_circular_uniform_to_moon_drift_flow_matching.pt
```

## Sample the saved model

Reuse that checkpoint to generate trajectories at additional NFE values:

```bash
dfm-trajectories \
  --load-dir outputs/demo/checkpoints \
  --nfe 1 2 5 20 --output outputs/resampled
```

The figures appear in `outputs/resampled/`. Checkpoints store the inference model
and training settings; DFM uses EMA weights. Keep the evaluation seed and sample
count unchanged to compare the same source points.

## Compare all four tasks

```bash
dfm-trajectories --config configs/trajectory_comparison.json --device cuda:0
```

This trains each of the 12 method/dataset combinations for 40,000 updates,
evaluating FM at NFE 50 and DFM/MeanFlow at NFE 1 and 20. CUDA execution uses a
matching PyTorch 2.8.0 CUDA build. The [protocol](docs/PROTOCOL.md) lists model,
drift, and training settings.

## Read the outputs

Each method/dataset produces a `.pt` checkpoint, `.png` figures, `.npz` traces,
and a `.json` record of the checkpoint hash, evaluation settings, dataset
parameters, and plot bounds.

In each figure, blue points are source samples, red points are generated
endpoints, and gray points represent the target distribution. Lines connect the
saved sampler states. All selected methods and NFE share the source points and
coordinate bounds for a given dataset and evaluation seed. `--max-display`
controls how many paths/source points are drawn; all generated endpoints and
target points are shown.

Each `.npz` contains `times` with shape `[NFE+1]`, `states` with shape
`[NFE+1, samples, 2]`, and `target_samples` with shape `[samples, 2]`.
`states[k]` is the sampler state at `times[k]`; target samples describe the
reference distribution. Each interval uses one network evaluation.

```python
import numpy as np
trace = np.load("outputs/demo/2d_circular_uniform_to_moon_drift_flow_matching_nfe_20.npz")
print(trace["states"].shape, trace["times"])
```

## Options and checks

`dfm-trajectories --help` lists the training, dataset, method, NFE, and device
options. JSON values can be overridden with CLI flags. DFM uses four time-pair
groups, so batch size must be a positive multiple of four. `--threads` sets CPU
intra-op threads (default: 1). `python -m flowviz` is an equivalent entrypoint.

```bash
# Two-update installation check.
dfm-trajectories --config configs/smoke.json

# Drift, training, checkpoint, trajectory, and CLI regression tests.
python -m unittest discover -s tests -v
```

## Code

The [training objectives](src/flowviz/pipelines/training.py) and
[samplers](src/flowviz/pipelines/inference.py) implement the three methods.
[Method settings](src/flowviz/configs/methods.py) and
[synthetic data](src/flowviz/data/synthetic.py) define the experiments.
