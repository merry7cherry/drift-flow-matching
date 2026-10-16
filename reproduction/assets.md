# Project-page figure and result sources

Figure mappings and checksums are in [provenance.json](../docs/assets/provenance.json). Table values and source hashes are in [results.json](../docs/assets/results.json).

## Figures

| Asset family | Manuscript source | Processing and meaning |
|---|---|---|
| `ffhq-nfe-{1,2,5,10}.jpg` | `img/optimized/result_FFHQ/ffhq_fake_steps_grid_step_{N}.pdf` | Complete-page raster exports, maximum dimension 1800 pixels; NFE 1, 2, 5 and 10. |
| `ffhq-{pca,umap}-nfe-{1,2,5,10}.jpg` | `img/optimized/img_FFHQ/`, included by `img/img_FFHQ.tex` | Eight unchanged JPGs. NFE follows the corresponding manuscript column caption; the source mapping is recorded below. Filled markers are generated latents and outlined markers are real latents. |
| `{dataset}-{method}.jpg` | `img/optimized/img_viz/2d_circular_uniform_to_{dataset}_{method}.jpg`, included by `img/img_viz.tex` | Unchanged optimized JPGs. The four datasets are moon, checkerboard_grid, letter_f and letter_m. Methods follow the manuscript labels: Flow Matching 50, Mean Flow 1/20, Drift Model 1, and DFM 1/20. |
| `{dataset}-ground_truth.jpg` | `img/optimized/img_viz/2d_circular_uniform_to_{dataset}_ground_truth.jpg`, included by `img/img_viz.tex` | Four unchanged first-column panels, labeled **Ground Truth** in the paper. Blue is the circular source distribution and red is the target distribution. |
| `group-{panel}.jpg` | `img/optimized/img_group_drift/`, included by `img/img_group_drift.tex` | Five unchanged grouped-drift illustration panels, mapped below. These show separate time-pair groups, not successive frames from a sampling trajectory. |

### FFHQ latent coverage

Figure `fig:ffhq_latent_inference_steps` in `img/img_FFHQ.tex` uses PCA in the top row and UMAP in the bottom row. Its column captions define NFE 1, 2, 5 and 10; the first column is the one-step Drift Model reference.

| Paper column | Paper NFE | Original PCA filename | Original UMAP filename |
|---|---|---|---|
| 1 | 1 | `latent_pca_real_fit_1.jpg` | `latent_umap_real_fit_1.jpg` |
| 2 | 2 | `latent_pca_real_fit_5.jpg` | `latent_umap_real_fit_5.jpg` |
| 3 | 5 | `latent_pca_real_fit_10.jpg` | `latent_umap_real_fit_10.jpg` |
| 4 | 10 | `latent_pca_real_fit_50.jpg` | `latent_umap_real_fit_50.jpg` |

### Grouped drift

The grouped-drift mapping preserves the manuscript panel order:

| Asset | Original filename | Meaning |
|---|---|---|
| `group-source-target.jpg` | `01_source_target.jpg` | Endpoint distributions: blue source `p_0`, red target `p_1` |
| `group-early.jpg` | `group_early_drift.jpg` | Group 1: `(t,r) = (0.05,0.30)` |
| `group-wide.jpg` | `group_wide_drift.jpg` | Group 2: `(t,r) = (0.20,0.80)` |
| `group-middle.jpg` | `group_middle_drift.jpg` | Group 3: `(t,r) = (0.35,0.65)` |
| `group-late.jpg` | `group_late_drift.jpg` | Group 4: `(t,r) = (0.70,0.95)` |

In the four time-pair panels, red denotes the target-time marginal, green denotes model output under the same model parameters, and dark arrows denote the grouped drift field.

## Quantitative data

`results.json` contains the following paper results:

- **MNIST and FFHQ:** the complete eight-row comparison in the left table of `tab/tab_combine.tex`, label `tab:mnist_ffhq_comparison`. MNIST uses latent squared 2-Wasserstein EMD and classifier accuracy (stored in percent). FFHQ uses latent EMD and FID against decoded real latents using the same frozen ALAE decoder.
- **ImageNet:** the four DFM L/2 rows in the right table of `tab/tab_combine.tex`, label `tab:image256`: NFE 1/2/5/10, FID 1.52/1.45/1.34/1.31 and IS 259.4/268.3/279.8/287.4. Evaluation uses 50,000 generated images at 256 × 256.
- **Robotics:** ToolHang State and PushT Visual rows from `tab/tab_robotic.tex`, label `tab:robotics_control`. Their success rates are 0.41/0.77/0.81/0.86 and 0.86/0.87/0.89/0.90, respectively, at NFE 1/2/5/10. Values are stored as fractions and average the last 10 checkpoints.

The evaluation protocols are described in `sec/05_experiment.tex`.

## Validation

From the repository root:

```bash
python scripts/check_site.py
```

This checks page references, figure mappings and hashes, NFE labels, and displayed values against `results.json`.

After changing result data, regenerate the static HTML blocks with:

```bash
python scripts/check_site.py --write-results
```

When the original manuscript checkout is available, use:

```bash
python scripts/check_site.py --manuscript-root /path/to/manuscript
```

This also checks original source hashes, figure inclusion, time-pair labels and values against the active LaTeX tables. An adjacent manuscript checkout is detected automatically.
