# Trajectory protocol

The four tasks use a uniform circular source and checkerboard, F, M, or two-moons
targets. DFM, Flow Matching (FM), and MeanFlow use three hidden MLP layers of width
128 with SiLU activations.

## Drift Flow Matching

The implementation follows **Appendix Algorithm 2**. It uses Euclidean cost
`C(x, y) = ||x - y||`, logits `-C / temperature`, uniform source/target marginals,
and alternating log-space Sinkhorn updates. Each plan is normalized by row before
computing `V = P_cross @ target - P_self @ prediction`.

| Setting | Value |
|---|---:|
| Time-pair groups | 4 |
| Positive / negative temperature | 0.01 / 0.01 |
| Sinkhorn iterations | 5 |
| Query logit-normal mean / std | -1.0 / 2.5 |
| Reference logit-normal mean / std | 1.0 / 2.5 |
| EMA decay | 0.999 |

One sampled pair is shared per group and sorted so `t <= r`. For independent
source/target endpoints, training constructs `x_t = (1-t)x_0 + t*x_1` and
`x_r = (1-r)x_0 + r*x_1`, then predicts `x_hat_r = x_t + (r-t)u(x_t, t, r-t)`.
The target is `stopgrad(x_hat_r + V)`. The loss averages the squared residual norm
across samples. Drift is computed separately within each group.

A group containing one source/target pair reduces to conditional endpoint
regression: its drift is `x_r - x_hat_r`. Larger groups estimate the marginal
transport signal. The default batch size is 128 (32 samples per group); the full
comparison uses 512 (128 per group).

Inference uses uniform intervals and the same two-time transport update. DFM and
MeanFlow share the two-time MLP architecture, with separate training objectives
and time conventions. FM learns instantaneous velocity and uses Euler sampling.

## Run settings

| Configuration | Updates per method/task | Purpose |
|---|---:|---|
| `configs/smoke.json` | 2 | Installation check |
| Default CLI | 1,000 | DFM two-moons demo |
| `configs/trajectory_comparison.json` | 40,000 | Three methods on four tasks |

The comparison uses 400 epochs, 100 steps per epoch, batch size 512, and Adam
with learning rate 0.001. It evaluates FM at NFE 50 and DFM/MeanFlow at NFE 1 and 20.
See the [module README](../README.md) for commands and output formats.
