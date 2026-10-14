# Generation Trajectory

A compact implementation of Drift Flow Matching (DFM), with Flow Matching and
MeanFlow baselines, on the four 2D synthetic distributions used in the paper:
checkerboard, letters F and M, and two moons. All data are generated in memory.

The default runs one small DFM experiment on CPU. It saves a model checkpoint,
actual sampler trajectories, and figures. No dataset download or cluster account
is required.

## Install

Use Python 3.11 or 3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package pins PyTorch, NumPy, and Matplotlib versions. For CUDA, install the
matching PyTorch 2.8.0 build for your machine before installing this package.
The same commands and relative output directories work on a normal workstation.

## Quick start

```bash
# One moon dataset, DFM, 1,000 optimizer updates, CPU; NFE 1 and 20.
dfm-trajectories --output outputs/demo

# Minimal execution check; not a trained-quality result.
dfm-trajectories --config configs/smoke.json
```

`python -m flowviz` is equivalent to `dfm-trajectories`. Useful flags include
`--device cpu` / `--device cuda:0`, `--epochs`, `--steps-per-epoch`,
`--batch-size`, `--lr`, `--nfe 1 2 5 20`, `--eval-samples`, `--seed`, and
`--output`. `--threads` controls CPU intra-op threads (default: 1).
DFM uses four time-pair groups: batch size must be a positive multiple of four.
A batch of four gives one sample per group (conditional regression); larger groups
estimate the distribution-level drift.
JSON config values can be overridden with CLI flags.

## Generate from a saved model

```bash
dfm-trajectories \
  --load-dir outputs/demo/checkpoints \
  --datasets 2d_circular_uniform_to_moon \
  --methods drift_flow_matching \
  --nfe 1 2 5 20 --output outputs/resampled
```

This loads the checkpoint and does **no training**. Each dataset/method pair uses
its own named `.pt` file. Checkpoints contain CPU tensor state dictionaries,
architecture, training configuration, seed, and loss history. DFM checkpoints
contain EMA inference weights; these files are for inference and replotting,
not optimizer-resume training. Use the same evaluation seed and sample count to
recover identical source points.

## Compare the paper's trajectory tasks

```bash
dfm-trajectories --config configs/trajectory_comparison.json --device cuda:0
```

This requests 40,000 updates **per method and dataset**. It is a full experiment,
not the quick start. The configuration retains the research code's recovered
training budget and method hyperparameters, with FM at NFE 50 and DFM/MeanFlow at
NFE 1 and 20.

See the [protocol](docs/PROTOCOL.md) for model and drift settings, and the
[release scope](../reproduction/README.md) for the relationship between new runs
and archived paper results.

## Outputs and their meaning

For each selected method/dataset:

- `checkpoints/<dataset>_<method>.pt`: inference model and training metadata.
- `<dataset>_<method>_nfe_<N>.npz`: `times` of shape `[N+1]`, `states` of shape
  `[N+1, samples, 2]`, and `target_samples` of shape `[samples, 2]`.
- Corresponding `.png`: blue source, red generated endpoint, gray target samples,
  and lines connecting the actual sampler states. Axes are fixed across all
  selected methods and NFE for each dataset. Only the selected `--max-display` paths/source points are drawn;
  all generated endpoints and target samples are shown.
- `<dataset>_<method>.json`: complete checkpoint metadata, evaluation settings,
  dataset parameters, checkpoint SHA-256, shared plot bounds, and trace filenames.

Target samples show a distribution. They are **not pathwise ground truth** for
the generated trajectories. All selected methods and NFE use the same source
samples for a given evaluation seed. A line between two saved states is only a
visual connection, not an additional network evaluation.

```python
import numpy as np
trace = np.load("outputs/demo/2d_circular_uniform_to_moon_drift_flow_matching_nfe_20.npz")
print(trace["states"].shape, trace["times"])
```

## Code and checks

The DFM objective is in `src/flowviz/pipelines/training.py`; the sampler is in
`src/flowviz/pipelines/inference.py`. Method hyperparameters are in
`src/flowviz/configs/methods.py`, and dataset definitions are in
`src/flowviz/data/synthetic.py`. The drift computation follows Appendix
Algorithm 2: grouped cross-minus-self Sinkhorn drift with Euclidean-distance
logits. DFM regresses a stopped-gradient state target.

```bash
python -m unittest discover -s tests -v
```

The tests cover drift invariances, single-sample groups, optimizer updates,
checkpoint round trips, matched-source exports, shared comparison axes,
load-without-training, and CLI validation.
