# Reproduction scope

This release provides runnable training, sampling, evaluation, and visualization workflows. Distinguish three kinds of evidence:

1. **Paper-reported results.** Values transcribed from the manuscript, and archived figures shown on the project page.
2. **Release workflow verification.** Small tests of installation, training, checkpoint reload, sampling, and evaluation. These do not establish the paper's numerical results.
3. **Full reproduction.** Re-running the matched dataset representation, preprocessing, model, seed, budget, checkpoint selection, and evaluation protocol. Original paper checkpoints and historical run manifests have not been recovered, so exact numerical reproduction is not claimed.

## Scientific configuration

The image module uses the paper's fixed-class grouped drift construction: a Sinkhorn-weighted positive transport term toward samples of the same class, minus the generated-sample term. The source-development variants with omega conditioning and cross-class mixing are omitted.

The trajectory module uses grouped DFM without the later self-consistency or auxiliary Flow Matching loss. It includes the four synthetic datasets used in the manuscript visualization: two moons, checkerboard, letter F, and letter M. Its retained Flow Matching and MeanFlow baselines support method comparisons. The archived Drift Model figures are shown for context; they do not imply a separate Drift Model training implementation in this release.

## Protocols

- [Image generation](image_generation.md)
- [Generation trajectories](generation_trajectory.md)
- [Archived figure sources](assets.md)
- [Release verification](VERIFICATION.md)

## Paper-reported image metrics

These values are transcribed from the active manuscript table `tab/tab_combine.tex`, not calculated by the release verification tests.

| Method | NFE | MNIST EMD | MNIST accuracy | FFHQ EMD | FFHQ FID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flow Matching | 50 | 37.2 | 100.0% | 198.4 | 77.4 |
| MeanFlow | 1 | 68.6 | 96.7% | 258.7 | 131.2 |
| MeanFlow | 2 | 48.1 | 99.1% | 211.7 | 85.4 |
| MeanFlow | 5 | 37.5 | 100.0% | 204.2 | 82.1 |
| Drift Model | 1 | 37.3 | 100.0% | 225.5 | 116.2 |
| DFM | 1 | 37.3 | 100.0% | 225.5 | 116.2 |
| DFM | 2 | 37.2 | 100.0% | 215.7 | 80.4 |
| DFM | 5 | 37.2 | 100.0% | 196.2 | 75.9 |

EMD denotes squared Wasserstein-2 distance in latent space. FFHQ FID compares generated latents decoded by the frozen ALAE decoder against real latents decoded by that same decoder; it is not FID against the original raw FFHQ images.

## Source history

The public release was prepared from these development snapshots:

- Image implementation: `e56388a9c587ce8b2d0637c5e73928f934244eb4`.
- Trajectory implementation: `bd163c5b9ccd8ea764b6fcd1870045cea388c280`.
- Manuscript and archived figures: `d676d9e13d6ec08cab8e3a39d063b5605e151f1c`.

These are provenance identifiers for the preparation sources, not claims that any one snapshot produced the original paper results. The public Git history records the portability and simplification changes.
