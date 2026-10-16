# Architecture

The image module has one training objective and one conditional latent MLP, with dataset-specific preprocessing and image evaluation.

| Layer | Responsibility |
|---|---|
| `config` | Typed YAML settings, explicit data/output roots, config-relative path normalization |
| `data` | MNIST `.npy` and FFHQ six-class `.npz` pools; balanced sampling |
| `architectures` | MLP accepting latent state, time, interval and class; MNIST AE; inference-only ALAE |
| `methods` | Same-class grouped Sinkhorn drift, stop-gradient matching target and NFE sampling |
| `training` | AdamW, EMA, logs, checkpoint save/resume and reproducible random state |
| `evaluation` | Dataset-specific metrics and optional visualization |
| `tasks` | Train, sample, evaluate, preprocessing and a small CPU smoke command |

Start with `tasks/train.py`, which loads the configuration, constructs the data, model and method, and runs the trainer. `tasks/sample.py` loads a generator checkpoint and its matching decoder for image output. `tasks/evaluate.py` evaluates a checkpoint against the dataset's reference split.
