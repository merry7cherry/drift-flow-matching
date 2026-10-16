# FFHQ image workflow

FFHQ uses a frozen ALAE encoder/decoder with RGB inputs at the checkpoint's resolution (standard FFHQ: 1024×1024) and a six-class latent generator. Run commands from `image_generation/` after installation.

## Prepare images and classes

Provide a CSV or JSONL manifest with `path`, plus either `class_name` or the attributes `gender` and `age`/`age_group`. An optional `split` field fixes train/test membership. Without it, the split command creates a deterministic per-class 90/10 split with seed 42.

```bash
driftfm-ffhq-build-split --manifest /path/to/manifest.csv \
  --image-root /path/to/images --output-dir data/ffhq_6class
```

The output is `train/<class>/*` and `test/<class>/*` for these six classes:

```text
male_children   male_adult   male_old
female_children female_adult female_old
```

The age groups are 0–19, 20–49, and 50+.

An optional helper supports the [FFHQ-Aging mirror](https://huggingface.co/datasets/NUS-SRI-2025/FFHQ-Aging-Dataset):

```bash
python -m pip install -e '.[data]'
driftfm-ffhq-download-aging --image-dir data/ffhq_images \
  --manifest-out data/ffhq_manifest.csv --work-dir data/ffhq_download
```

Use `data/ffhq_manifest.csv` and `data/ffhq_images` as the manifest and image root in the split command above.

## Import ALAE and encode latents

Obtain the FFHQ configuration and checkpoint from the [official ALAE repository](https://github.com/podgorskiy/ALAE), then:

```bash
driftfm-ffhq-import-alae --alae-root /path/to/ALAE \
  --output-dir runs/ffhq_alae/imported

driftfm-ffhq-encode-latents \
  --alae-ckpt runs/ffhq_alae/imported/alae_ffhq.pt \
  --image-root data/ffhq_6class \
  --output-dir data/ffhq_latents_6class --device cuda --batch-size 8
```

The import reads `configs/ffhq.yaml` and the checkpoint referenced by `training_artifacts/ffhq/last_checkpoint`, and saves `alae_ffhq.pt`.

Encoding writes `train_latents_by_class.npz` and `test_latents_by_class.npz`. Each contains the six class keys above, with a floating-point `(N,D)` array per class. The standard ALAE FFHQ representation uses `D=512`.

## Train and generate

```bash
driftfm-train --config configs/experiments/ffhq_driftfm.yaml

driftfm-sample --checkpoint runs/ffhq_drift/ffhq_dfm/checkpoint_final.pt \
  --decoder-checkpoint runs/ffhq_alae/imported/alae_ffhq.pt \
  --output-dir runs/ffhq_samples --steps 1 2 5 --device cuda --batch-size 8

driftfm-evaluate --checkpoint runs/ffhq_drift/ffhq_dfm/checkpoint_final.pt \
  --config configs/experiments/ffhq_eval.yaml
```

The supplied generator configuration uses batch size 3,072 and 200 epochs of 100 optimizer steps. Use the same frozen ALAE checkpoint for encoding, image sampling and evaluation.

## Evaluation semantics

The standalone evaluation reports latent OT and FID at NFE 1, 2 and 5. Latent OT compares generated samples with held-out test latents using EMD with squared Euclidean cost (`W₂²`), averaged across classes. Add `pca` and `umap` to `metrics` in `ffhq_eval.yaml` to generate the paper's latent visualizations.

FID compares generated samples with reconstructions of held-out real latents, decoded by the same frozen ALAE model with decoder noise enabled. The sample command uses deterministic decoding for its image grids.

Evaluation saves one directory per NFE and an `evaluation_index.json`.
