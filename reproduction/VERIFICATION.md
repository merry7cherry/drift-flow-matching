# Release verification

Checked on 2026-10-04. This record describes software verification, not a rerun of the paper's full experiments.

## Clean installation and CPU execution

Both packages were installed together in a fresh Python 3.11.15 environment on macOS arm64 with PyTorch 2.8.0, torchvision 0.23.0, NumPy 2.0.2, and Matplotlib 3.9.4. The image module's `constraints-tested.txt` records these core versions. The GitHub Actions workflow repeats installation and tests on Ubuntu; consult the repository's Actions history for the current Linux result.

- Image module: **130 tests passed**, including grouped drift math, class labels, preprocessing, checkpoint resume/RNG recovery, portable evaluation paths, all CLI help entrypoints, and actual MNIST/ALAE decoding using small random test models. Multiprocess DataLoader tests require the operating system's normal shared-memory permissions.
- Trajectory module: **5 tests passed**, including optimizer updates for DFM/FM/MeanFlow, checkpoint round trips, matched source samples, and load-only execution.
- The installed `driftfm-smoke` command ran from outside the source checkout. It trained on synthetic latents, saved/reloaded an EMA checkpoint, sampled NFE 1/2, and calculated latent OT.
- The installed `dfm-trajectories` command trained the supplied two-update smoke configuration from outside the source checkout. A subsequent load-only invocation produced exactly equal `times`, `states`, and `target_samples` arrays at NFE 1/2.

## Cleanup equivalence

The original class-only image implementation (`split_v0`) and the released implementation were compared on the same fixed CPU fixture. Loss, all 524 parameter-gradient entries, grouped Sinkhorn drift, and sampled outputs at NFE 1/2/5 were exactly equal. DFM, FM, and MeanFlow trajectory implementations were also compared against the recovered source on a four-update fixture: losses and final parameters matched exactly. These are bounded regression checks, not proof for every possible configuration.

## Project page

All 28 archived figure assets were checked against their provenance hashes. Local asset links and page anchors passed automated checks. Desktop and 390-pixel mobile layouts, NFE switching, distribution/method selectors, and BibTeX copying were checked in a browser. The controls display archived paper figures; they do not perform model inference.

## Not verified by these checks

Full MNIST/FFHQ training, full-budget synthetic comparisons, CUDA execution, and exact reproduction of historical paper metrics remain outside this verification. The original paper checkpoints and historical run manifests are not included. See the [reproduction scope](README.md) and dataset-specific protocols before comparing results.
