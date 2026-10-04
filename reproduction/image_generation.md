# Image generation protocol

Use the [module README](../image_generation/README.md) for exact installation and command examples.

## MNIST

1. Obtain the standard MNIST train and test splits.
2. Train the supplied autoencoder on training images, then freeze it.
3. Encode training and test images with that same encoder. Keep split identity explicit.
4. Train a class-conditioned DFM model on the training latents.
5. Sample with the trained model at each reported NFE; use a fixed initial noise pool for paired step comparisons.
6. Compare generated latents with held-out test latents using the documented squared-L2 EMD protocol. Classification accuracy requires a classifier trained on training data.

Retain the exact autoencoder, classifier, generator checkpoint, resolved config, seed, package versions, device, and sample count with each evaluation. Changing the latent representation changes the scale of EMD.

## FFHQ

1. Obtain FFHQ images and the metadata required for the documented six-class split under their provider terms.
2. Import the official compatible ALAE model and keep it frozen.
3. Prepare train/test class splits and encode both with the same ALAE encoder.
4. Train DFM on the training latent pools; evaluate against held-out test pools.
5. Decode both generated and real latent samples using the same ALAE decoder and its documented noise setting.
6. Evaluate latent squared-L2 EMD and decoded-image FID at the chosen NFE values.

Record split membership, preprocessing, decoder checkpoint, decoder noise setting, model checkpoint, seed, number of generated/real samples, and FID implementation. A raw-image reference FID is a different metric and should be labeled separately.

## Available evidence

Public configs implement the manuscript's method and the available source settings. They are not recovered historical training manifests. The original image checkpoints are absent, and their training/selection provenance remains unavailable. Small CPU workflow checks do not establish full-data quality, CUDA behavior, or the paper's numbers. Reproducing full benchmarks requires supplying the datasets and completing the documented training and evaluation.
