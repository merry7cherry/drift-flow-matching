# Trajectory protocol and provenance

This module was extracted from the research trajectory implementation at
`bd163c5b9ccd8ea764b6fcd1870045cea388c280`. The public release retains only the four
2D tasks shown in the manuscript and its implemented FM, MeanFlow, and DFM
comparators. It removes deployment files, old compatibility exports, unrelated
synthetic tasks, Improved MeanFlow, Rectified Flow, and auxiliary training losses.

## Scientific settings

The manuscript describes the four target distributions, circular source, and
three-layer MLP. Figure comparisons use FM NFE 50 and DFM/MeanFlow NFE 1 and 20.
The source uses three hidden layers of width 128 with SiLU activations.

Recovered DFM implementation settings:

| Setting | Value |
|---|---:|
| Groups | 4 |
| Positive / negative temperature | 0.01 / 0.01 |
| Sinkhorn iterations | 5 |
| Query logit-normal mean / std | -1.0 / 2.5 |
| Reference logit-normal mean / std | 1.0 / 2.5 |
| EMA decay | 0.999 |
| Auxiliary FM / self-consistency loss | Absent |

The two sampled times are sorted so `t <= r`; one pair is shared per group.
Training interpolates paired source and target samples to these times, predicts
`x_r = x_t + (r - t) u(x_t, t, r - t)`, and matches a detached target formed by
adding cross-minus-self Sinkhorn drift. Inference uses uniform time intervals
and the same two-time update. One interval costs one network evaluation.

The research source default enabled a self-consistency extension. It was removed
because the manuscript's method and algorithm do not include that loss. This is
an explicit alignment to the stated method, not a claim to have identified the
exact code/configuration that produced the archived figures.

## Three levels of use

| Protocol | Budget | What it verifies |
|---|---:|---|
| `configs/smoke.json` | 2 optimizer updates | Installation, train/save/trace/render execution; load is tested separately |
| Default CLI demo | 1,000 updates | Small independent DFM demonstration |
| `configs/trajectory_comparison.json` | 40,000 updates per method/task | New comparison with the recovered research budget |

The comparison configuration's 400 epochs, 100 steps/epoch, batch size 512, Adam
learning rate 0.001, and numeric method settings come from the recovered source.
They were not stated as a complete synthetic training protocol in the manuscript.
Historical commands, seeds, and checkpoint lineage remain unverified. Therefore
no configuration is named or represented as an exact original-figure reproducer.

The one-step Drift Models comparison in the manuscript is not part of this source
snapshot. Source images from the paper may be presented as archived paper results,
but newly generated outputs must be identified as new runs.

## Interpreting results

A smoke run is not a quality benchmark. Increasing inference NFE is not guaranteed
to improve every newly trained toy checkpoint. Assess the actual distribution and
trajectories; do not relabel training interpolation paths as model trajectories.
Checkpoint manifests retain the exact settings and weights hash used for each
new export. Hardware and runtime should accompany any reported training costs.
