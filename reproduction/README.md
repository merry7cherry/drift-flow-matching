# Reproduction guide

This release contains training, sampling, evaluation, and visualization workflows for MNIST, FFHQ, and four synthetic datasets. Pretrained DFM checkpoints are not distributed; train your own model using the workflows below. FFHQ uses the documented frozen ALAE assets. ImageNet and robotics implementations are outside this release.

## Workflows

- [Image generation](image_generation.md): prepare data, train, sample and evaluate MNIST / FFHQ.
- [Generation trajectories](generation_trajectory.md): train DFM / FM / MeanFlow, reload checkpoints and compare trajectories.
- [Figure sources](assets.md): original paper figures, panel labels and quantitative data.
- [Software verification](VERIFICATION.md): tested environments and regression results.

## Method and experimental setup

Both modules implement the paper's **Appendix Algorithm 2**: grouped Sinkhorn drift with Euclidean-distance logits, balanced transport weights, and row normalization. This is the concrete drift-field choice for the general distribution-matching objective in the main text. Each group shares a time pair; image-generation groups also share a class. The trajectory module retains Flow Matching and MeanFlow for the paper's method comparisons.

The image workflows use a 16-dimensional MNIST autoencoder and frozen 512-dimensional FFHQ ALAE latents. EMD is the average per-class squared Wasserstein-2 distance against held-out test latents. FFHQ FID compares generated and real latents decoded with the same ALAE decoder. The dataset guides specify preprocessing and evaluation settings.

## Paper results and reproduction

The [project page](https://merry7cherry.github.io/drift-flow-matching/#results) presents the paper's original figures and table values. Its controls browse those figures. NFE labels and Ground Truth panels follow the paper; source filenames and checksums are recorded in the figure manifest.

Public training configurations were reconstructed from the available research source and manuscript. Original experiment checkpoints and complete historical run manifests are unavailable, including verified seed pairing for the archived image grids and trajectories. Full numerical reproduction therefore requires matching preprocessing, representation, model, seed, budget, checkpoint selection and evaluation. Software tests cover execution and regressions; they do not replace full benchmark training.
