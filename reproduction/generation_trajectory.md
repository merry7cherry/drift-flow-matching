# Generation trajectory protocol

The [module README](../generation_trajectory/README.md) gives installation,
training, checkpoint reuse, and output commands. The
[protocol](../generation_trajectory/docs/PROTOCOL.md) specifies the grouped
Sinkhorn implementation from Appendix Algorithm 2 and all run settings.

To compare inference budgets, train once for each method/dataset, then reuse its
checkpoint and the same evaluation seed and sample count at every NFE. Export the
actual sampler states and plot every method/NFE with the same coordinate bounds.
The supplied comparison command performs this workflow on the four paper tasks.

The `.npz` files contain sampler states and times, plus target-distribution samples.
Target samples do not define a unique ground-truth path. The accompanying `.json`
files retain resolved settings, checkpoint hashes, and plot bounds.

See [release scope and reproduction](README.md) for the coverage of this release
and how its new runs relate to the archived paper figures.
