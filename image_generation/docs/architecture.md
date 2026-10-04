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

No site-specific runtime is required. Relative config paths use the configuration file as their anchor; relative command-line paths use the working directory. Configurations stored in checkpoints contain resolved absolute paths. Independent sampling only requires a generator checkpoint and, for image output, its matching decoder. Evaluation additionally requires the appropriate reference data.

The image and generation-trajectory modules are separately installable. Neither module needs the other's environment or data.
