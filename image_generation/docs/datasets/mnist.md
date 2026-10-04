# MNIST image workflow

MNIST uses a convolutional autoencoder and a class-conditional latent MLP. The following is a complete training workflow; it is not a pretrained-model quickstart. Run commands from `image_generation/` after installation with the `eval` extra.

## Prepare the encoder and latent data

```bash
driftfm-mnist-train-ae --download --data-dir data/mnist \
  --output-dir runs/mnist_ae --latent-dim 16 --epochs 200 --device cuda

driftfm-mnist-encode-latents --data-dir data/mnist \
  --ae-ckpt runs/mnist_ae/latest/ae_final.pt --device cuda
```

`--download` explicitly permits MNIST acquisition through torchvision. Omit it when the data is already present. Change the training budget/device for a local software check as needed; doing so changes the experiment. The paper specifies a 16-dimensional MNIST representation. The 200-epoch encoder schedule is recovered from the preprocessing CLI; its exact historical checkpoint is not available.

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

Keep the concrete autoencoder checkpoint fixed between encoding, training, image sampling and evaluation. A different latent dimension or encoder state defines a different representation. The generator architecture derives its latent dimension from the data.

The default metrics are latent OT, decoded image OT, and classifier accuracy. Evaluation uses the test latent pool. Image OT is computed on flattened image pixels with the configured distance/solver. If the configured MNIST classifier checkpoint is absent, the evaluator trains its auxiliary classifier using the locally staged MNIST training data and saves it. For a comparable result, record and reuse that exact classifier.

An evaluation-only YAML can select `metrics: [latent_ot]` and a smaller `n_per_class`, but the MNIST evaluator still loads the matching decoder to write its class grid. `pca`, `tsne`, `umap` and `lda` remain optional diagnostic metrics and are excluded from the default result sweep.

Image checkpoints for the reported paper results are not bundled. Exact historical reproduction additionally needs the original encoder, generator, classifier, dataset split, seeds, budgets and metric settings.
