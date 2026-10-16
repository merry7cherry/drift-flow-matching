# Drift Flow Matching implementation

DFM uses the paper's fixed-class drift construction. The conditional MLP predicts `u(x_t, t, h, c)` for class `c`, where `h = r - t` represents the interval between the ordered times `(t, r)`.

For a class-balanced batch, the method partitions each class into `groups_per_class` subgroups. A subgroup index shares a sampled pair `0 <= t <= r <= 1` across classes, while each transport problem uses samples from one class only. Time values follow the configured logit-normal laws.

For each group:

1. Form `x_t = (1-t) x0 + t x1` and `x_r = (1-r) x0 + r x1` from Gaussian noise `x0` and real latent data `x1`.
2. Predict `y = x_t + (r-t) u(x_t,t,r-t,c)`.
3. Compute a log-space Sinkhorn barycentric projection from `y` to `x_r`, and another from `y` to `y`, using uniform marginals.
4. Let `drift = projection(y,x_r) - projection(y,y)`.
5. Regress `y` to the stop-gradient target `y + drift`.

This is the grouped Sinkhorn implementation in the paper's Appendix **Algorithm 2**: kernel logits are `-torch.cdist(source, target) / temperature` (Euclidean distance), with 20 Sinkhorn iterations in the supplied image configurations and row normalization of the resulting plan. The loss is squared Euclidean error, summed over latent coordinates and averaged over samples.

Sampling uses `N` uniform intervals from 0 to 1 and applies `x_next = x + h u(x,t,h,c)` once per interval. NFE therefore equals `N`. Generation uses explicit class labels.
