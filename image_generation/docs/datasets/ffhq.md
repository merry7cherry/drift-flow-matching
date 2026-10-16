# FFHQ image workflow

FFHQ uses a frozen ALAE encoder/decoder and a six-class latent generator. This workflow requires FFHQ images, class metadata and the official ALAE checkpoint. Run commands from `image_generation/` after installation with the `eval` extra.

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

The age groups are 0–19, 20–49, and 50+. The command saves the manifest and split metadata, and creates symlinks by default. Use `--materialize-mode copy` for a self-contained dataset.

An optional helper supports the [FFHQ-Aging mirror](https://huggingface.co/datasets/NUS-SRI-2025/FFHQ-Aging-Dataset):

```bash
python -m pip install -e '.[data]'
driftfm-ffhq-download-aging --image-dir data/ffhq_images \
  --manifest-out data/ffhq_manifest.csv --work-dir data/ffhq_download
```

Pass the generated manifest to the split command above. Downloaded ZIP archives are deleted after successful extraction; use `--keep-archives` to retain them.

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

The import reads `configs/ffhq.yaml` and `training_artifacts/ffhq/last_checkpoint` with its referenced checkpoint, or accepts explicit `--official-config` and `--official-checkpoint` paths. It saves `alae_ffhq.pt` and updates the local `latest` marker used by the training and evaluation configurations.

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

The default metrics are latent OT and FID at NFE 1, 2 and 5. Latent OT compares generated samples with held-out test latents using EMD with squared Euclidean cost (`W₂²`), averaged across classes. Select `metrics: [latent_ot, fid, pca, umap]` in `ffhq_eval.yaml` to add the paper's latent visualizations.

FID compares generated samples with reconstructions of held-out real latents, decoded by the same frozen ALAE model. Decoder noise is enabled for both by default. Use `--no-fid-decode-noise` with `driftfm-eval-ffhq` to disable it. The sample command uses deterministic decoding for its image grids.

Evaluation saves one directory per NFE and an `evaluation_index.json`. For latent-only evaluation, select `metrics: [latent_ot]` or run:

```bash
driftfm-eval-ffhq --checkpoint /path/to/generator.pt \
  --real-npz /path/to/test_latents_by_class.npz \
  --metrics latent_ot --num-sampling-steps 1 2 5 --device cpu
```

A full FID run may download the evaluator's Inception weights.
