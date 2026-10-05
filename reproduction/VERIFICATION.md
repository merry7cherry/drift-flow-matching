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

The paper-aligned page revision was checked on 2026-10-05 UTC (2026-10-04 America/New_York). All **37 archived figure assets** match their provenance hashes: 28 trajectory/reference panels, five grouped-drift panels, and four FFHQ grids. Local page references, anchors, JavaScript syntax, time-pair labels, and all entries in `docs/assets/results.json` passed verification. The results data was compared with the active manuscript tables; all displayed tables received an independent scientific review.

Browser checks covered the default DFM 1/20-NFE comparison, independent DFM NFE selection, method-specific NFE options, all three comparison presets, all four target distributions, four training time pairs, FFHQ image/metric switching (including the explicit missing 10-NFE metrics), and BibTeX copying. Desktop and 390-pixel mobile layouts were inspected; neither had page-level horizontal overflow and no JavaScript errors were observed. Tables scroll within their own regions on small screens.

The controls display archived paper figures, not live model inference. NFE counts function evaluations rather than wall-clock cost. ImageNet and robotics are shown as paper evidence and remain outside the initial implementation release. The page does not certify historical paired seeds or checkpoint identity.

## Not verified by these checks

Full MNIST/FFHQ training, full-budget synthetic comparisons, CUDA execution, and exact reproduction of historical paper metrics remain outside this verification. The original paper checkpoints and historical run manifests are not included. See the [reproduction scope](README.md) and dataset-specific protocols before comparing results.
