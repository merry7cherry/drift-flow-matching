# FFHQ image workflow

FFHQ uses a frozen ALAE encoder/decoder and a six-class latent generator. ALAE training is outside this release. Obtain the dataset and official ALAE artifacts under their respective terms; no data or pretrained weights are included here.

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

The inherited grouping uses ages 0–19, 20–49, and 50+; these are data preprocessing categories. The manifest and split metadata are retained. The default materialization uses symlinks; use `--materialize-mode copy` when moving a self-contained dataset to another machine.

An optional helper supports the [FFHQ-Aging mirror](https://huggingface.co/datasets/NUS-SRI-2025/FFHQ-Aging-Dataset):

```bash
python -m pip install -e '.[data]'
driftfm-ffhq-download-aging --image-dir data/ffhq_images \
  --manifest-out data/ffhq_manifest.csv --work-dir data/ffhq_download
```

This can download a large dataset and is never invoked by the smoke check or training command. The generated manifest can be passed to the split command above. The helper deletes its downloaded ZIP archives after successful extraction unless `--keep-archives` is supplied.

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

The import reads `configs/ffhq.yaml` and `training_artifacts/ffhq/last_checkpoint` with its referenced checkpoint, or accepts explicit `--official-config` and `--official-checkpoint` paths. It stores a self-contained inference checkpoint for this package. ALAE-derived implementation files retain the upstream Apache 2.0 notice.

The encoder checks both splits, class folders, RGB channels and image size against the imported model. Each output NPZ contains the six keys above, with a floating-point `(N,D)` array per class. The standard ALAE FFHQ representation uses `D=512`.

## Train and generate

Set `evaluation.params.alae_ckpt` in both FFHQ configs to `../../runs/ffhq_alae/imported/alae_ffhq.pt` (relative to those YAML files), or supply the absolute checkpoint path. Then:

```bash
driftfm-train --config configs/experiments/ffhq_driftfm.yaml

driftfm-sample --checkpoint runs/ffhq_drift/ffhq_dfm/checkpoint_final.pt \
  --decoder-checkpoint runs/ffhq_alae/imported/alae_ffhq.pt \
  --output-dir runs/ffhq_samples --steps 1 2 5 --device cuda --batch-size 8

driftfm-evaluate --checkpoint runs/ffhq_drift/ffhq_dfm/checkpoint_final.pt \
  --config configs/experiments/ffhq_eval.yaml
```

The full configuration uses batch size 3,072 and 200 epochs of 100 optimizer steps. Hardware requirements and runtime have not been measured for this public release. A small CPU test is available through `driftfm-smoke`; it uses four-dimensional synthetic data, not FFHQ images.

## Evaluation semantics

The default metrics are per-class latent OT and FID. The reference pool is explicitly the held-out test latent NPZ. There is no fallback to training data if that test file is missing.

**The inherited FID compares decoded generated latents against ALAE reconstructions of the real test latents, not raw original FFHQ images.** Both distributions use the same frozen decoder. Its result is not interchangeable with raw-image-reference FFHQ FID. `fid_decode_noise: true` in the evaluation config enables decoder noise on both real and generated images, with recorded seed; the sample CLI's presentation grids use `noise=False`.

Real decoded image caches are reused only when the recorded source/decoder/seed/settings and actual image counts match. Each NFE has separate generated-image artifacts. For latent-only evaluation, remove `fid` from an evaluation configuration or run:

```bash
driftfm-eval-ffhq --checkpoint /path/to/generator.pt \
  --real-npz /path/to/test_latents_by_class.npz \
  --metrics latent_ot --num-sampling-steps 1 2 5 --device cpu
```

A full FID run may download the evaluator's Inception weights. Exact historical table reproduction remains pending the original split, ALAE and generator checkpoints, evaluation settings and run records.
