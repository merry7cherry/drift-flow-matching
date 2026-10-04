# Drift Flow Matching implementation

The public implementation uses the paper's fixed-class drift construction. Each conditional MLP receives `u(x_t, t, h, class_labels)` with interval length `h = r - t`. This is an equivalent coordinate system for the ordered time pair `(t, r)`.

For a class-balanced batch, the method partitions each class into `groups_per_class` subgroups. A subgroup index shares a sampled pair `0 <= t <= r <= 1` across classes, while each transport problem uses samples from one class only. Time values follow the configured logit-normal laws.

For each group:

1. Form `x_t = (1-t) x0 + t x1` and `x_r = (1-r) x0 + r x1` from Gaussian noise `x0` and real latent data `x1`.
2. Predict `y = x_t + (r-t) u(x_t,t,r-t,c)`.
3. Compute a log-space Sinkhorn barycentric projection from `y` to `x_r`, and another from `y` to `y`, using uniform marginals.
4. Let `drift = projection(y,x_r) - projection(y,y)`.
5. Regress `y` to the stop-gradient target `y + drift`.

The kernel logits use negative Euclidean distance divided by the configured temperature. The plan is row-normalized before the barycentric projection. `norm_p=0`, as supplied, gives mean squared Euclidean error summed over latent coordinates. EMA is maintained by the trainer and used for sampling/evaluation.

Sampling uses `N` uniform intervals from 0 to 1 and applies `x_next = x + h u(x,t,h,c)` once per interval. NFE therefore equals `N`. Generation uses explicit class labels.

The source research branch also contained guidance experiments. Those variants and their configurations are excluded from this release. Their checkpoints are incompatible with this class-only model. The name `split_v0` remains in configuration as a provenance identifier for the retained formulation.

Scientific mapping does not establish numerical result provenance: complete original image checkpoints, encoder states, exact data splits and run records are still needed to reproduce the reported tables.
