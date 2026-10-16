# MNIST image workflow

MNIST uses a 16-dimensional convolutional autoencoder and a class-conditional latent MLP. Run commands from `image_generation/` after installation.

## Prepare the encoder and latent data

```bash
driftfm-mnist-train-ae --download --data-dir data/mnist \
  --output-dir runs/mnist_ae --latent-dim 16 --epochs 200 --device cuda

driftfm-mnist-encode-latents --data-dir data/mnist \
  --ae-ckpt runs/mnist_ae/latest/ae_final.pt --device cuda
```

Latent encoding writes `train_latents.npy`, `train_labels.npy`, `test_latents.npy`, and `test_labels.npy` beside `ae_final.pt`. Latents have shape `(N,16)` and labels are integers in `0..9`.

## Train, sample and evaluate

```bash
driftfm-train --config configs/experiments/mnist_driftfm.yaml

driftfm-sample --checkpoint runs/mnist_drift/mnist_dfm/checkpoint_final.pt \
  --decoder-checkpoint runs/mnist_ae/latest/ae_final.pt \
  --output-dir runs/mnist_samples --steps 1 2 5 --device cuda

driftfm-evaluate --checkpoint runs/mnist_drift/mnist_dfm/checkpoint_final.pt \
  --config configs/experiments/mnist_eval.yaml
```

The supplied generator configuration uses batch size 16,000 and 200 epochs of 100 optimizer steps. Use the same autoencoder checkpoint for encoding, image sampling and evaluation.

The standalone evaluation reports latent OT, image OT, and classifier accuracy at NFE 1, 2 and 5. Latent OT compares generated and held-out test latents. Image OT compares decoded generated images with original MNIST test images in flattened pixel space. Both use EMD with squared Euclidean cost (`W₂²`), averaged across classes. Evaluation trains a classifier on the local MNIST training split if needed and saves it for subsequent runs.

Add `umap` to `metrics` in `mnist_eval.yaml` to generate the paper's latent visualization.

Evaluation saves one directory per NFE and an `evaluation_index.json`.
