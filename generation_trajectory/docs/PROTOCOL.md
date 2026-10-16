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

Inference uses uniform intervals and the same two-time transport update. DFM and
MeanFlow share the two-time MLP architecture, with separate training objectives
and time conventions. FM learns instantaneous velocity and uses Euler sampling.

## Run settings

The default demo uses batch size 128 (32 samples per group). The full comparison
uses 400 epochs, 100 steps per epoch, batch size 512 (128 per group), and Adam
with learning rate 0.001. It evaluates FM at NFE 50 and DFM/MeanFlow at NFE 1 and 20.
See the [module README](../README.md) for commands and output formats.
