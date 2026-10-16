# MNIST image workflow

MNIST uses a convolutional autoencoder and a class-conditional latent MLP. The following workflow trains the encoder and generator. Run commands from `image_generation/` after installation with the `eval` extra.

## Prepare the encoder and latent data

```bash
driftfm-mnist-train-ae --download --data-dir data/mnist \
  --output-dir runs/mnist_ae --latent-dim 16 --epochs 200 --device cuda

driftfm-mnist-encode-latents --data-dir data/mnist \
  --ae-ckpt runs/mnist_ae/latest/ae_final.pt --device cuda
```

Omit `--download` when MNIST is already present. The autoencoder uses the paper's 16-dimensional representation and trains for 200 epochs.

The encoder writes `ae_final.pt`; latent encoding writes `train_latents.npy`, `train_labels.npy`, `test_latents.npy`, and `test_labels.npy` next to it. Latents are floating-point arrays `(N,D)`, labels are integer arrays `(N,)` in `0..9`. Every class must have samples.

## Train, sample and evaluate

```bash
driftfm-train --config configs/experiments/mnist_driftfm.yaml

driftfm-sample --checkpoint runs/mnist_drift/mnist_dfm/checkpoint_final.pt \
  --decoder-checkpoint runs/mnist_ae/latest/ae_final.pt \
  --output-dir runs/mnist_samples --steps 1 2 5 --device cuda

driftfm-evaluate --checkpoint runs/mnist_drift/mnist_dfm/checkpoint_final.pt \
  --config configs/experiments/mnist_eval.yaml
```

The supplied generator configuration uses batch size 16,000 and 200 epochs of 100 optimizer steps. Use the same autoencoder checkpoint for encoding, image sampling and evaluation. The generator derives its latent dimension from the data.

The default metrics are latent OT, image OT, and classifier accuracy at NFE 1, 2 and 5. Latent OT compares generated and held-out test latents. Image OT compares decoded generated images with original MNIST test images in flattened pixel space. Both use EMD with squared Euclidean cost (`W₂²`), averaged across classes. If the classifier checkpoint is absent, evaluation trains and saves a classifier using the local MNIST training data. Reuse that classifier for comparisons.

Select `metrics: [latent_ot, image_ot, accuracy, umap]` in `mnist_eval.yaml` to add the paper's MNIST latent visualization. PCA is also available. `metrics: [latent_ot]` selects only latent OT; evaluation also loads the matching decoder to write its class grid.

Evaluation saves one directory per NFE and an `evaluation_index.json`.
