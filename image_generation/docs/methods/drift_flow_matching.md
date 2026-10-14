# Drift Flow Matching implementation

The public implementation uses the paper's fixed-class drift construction. Each conditional MLP receives `u(x_t, t, h, conditioning)`, where `h = r - t` and `conditioning.class_labels` contains the class indices. This is an equivalent coordinate system for the ordered time pair `(t, r)`.

For a class-balanced batch, the method partitions each class into `groups_per_class` subgroups. A subgroup index shares a sampled pair `0 <= t <= r <= 1` across classes, while each transport problem uses samples from one class only. Time values follow the configured logit-normal laws.

For each group:

1. Form `x_t = (1-t) x0 + t x1` and `x_r = (1-r) x0 + r x1` from Gaussian noise `x0` and real latent data `x1`.
2. Predict `y = x_t + (r-t) u(x_t,t,r-t,c)`.
3. Compute a log-space Sinkhorn barycentric projection from `y` to `x_r`, and another from `y` to `y`, using uniform marginals.
4. Let `drift = projection(y,x_r) - projection(y,y)`.
5. Regress `y` to the stop-gradient target `y + drift`.

This is the grouped Sinkhorn implementation in the paper's Appendix **Algorithm 2**: kernel logits are `-torch.cdist(source, target) / temperature` (Euclidean distance), with 20 Sinkhorn iterations in the supplied image configurations and row normalization of the resulting plan. The paper's introductory row-normalized Gibbs drift and default squared cost describe a different concrete kernel recipe; they should not be substituted for Algorithm 2 when running these configurations. `norm_p=0`, as supplied, gives mean squared Euclidean error summed over latent coordinates. EMA is maintained by the trainer and used for sampling/evaluation.

Sampling uses `N` uniform intervals from 0 to 1 and applies `x_next = x + h u(x,t,h,c)` once per interval. NFE therefore equals `N`. Generation uses explicit class labels.

The v0.1 configuration names `drift_form: split_v0` and `conditioning_mode: class` are accepted only by the configuration/checkpoint compatibility reader. New configurations use the single class-conditional formulation directly, with the same parameter names and tensor shapes in saved model weights.

See the [reproduction guide](../../../reproduction/image_generation.md) for training recipes and the [verification record](../../../reproduction/VERIFICATION.md) for the scope of released results.
