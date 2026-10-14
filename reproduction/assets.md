# Project-page figure and result sources

The project page uses archived manuscript figures and the values reported in its active tables. No model outputs were generated, edited or improved for this page. The complete source-to-asset manifest and checksums are in [provenance.json](../docs/assets/provenance.json); quantitative data and its source hashes are in [results.json](../docs/assets/results.json).

## Figures

| Asset family | Manuscript source | Processing and meaning |
|---|---|---|
| `ffhq-nfe-{1,2,5,10}.jpg` | `img/optimized/result_FFHQ/ffhq_fake_steps_grid_step_{N}.pdf` | Existing complete-page raster exports, maximum dimension 1800 pixels. Captions and embedded titles agree with these NFE values. Selection and seed pairing are not recoverable; these are not established paired-seed comparisons. |
| `ffhq-{pca,umap}-nfe-{1,2,5,10}.jpg` | `img/optimized/img_FFHQ/`, included by `img/img_FFHQ.tex` | Eight unchanged JPGs. NFE follows the corresponding manuscript column caption; the source mapping is recorded below. Filled markers are generated latents and outlined markers are real latents. |
| `{dataset}-{method}.jpg` | `img/optimized/img_viz/2d_circular_uniform_to_{dataset}_{method}.jpg`, included by `img/img_viz.tex` | Unchanged optimized JPGs. The four datasets are moon, checkerboard_grid, letter_f and letter_m. Methods follow the manuscript labels: Flow Matching 50, Mean Flow 1/20, Drift Model 1, and DFM 1/20. |
| `{dataset}-ground_truth.jpg` | `img/optimized/img_viz/2d_circular_uniform_to_{dataset}_ground_truth.jpg`, included by `img/img_viz.tex` | Four unchanged first-column panels, labeled **Ground Truth** in the paper. Blue is the circular source distribution and red is the target distribution. |
| `group-{panel}.jpg` | `img/optimized/img_group_drift/`, included by `img/img_group_drift.tex` | Five unchanged grouped-drift illustration panels, mapped below. These show separate time-pair groups, not successive frames from a sampling trajectory. |

### FFHQ latent coverage

The active figure `fig:ffhq_latent_inference_steps` in `img/img_FFHQ.tex` defines the presentation: PCA is the top row, UMAP the bottom row, and the four columns are labeled NFE 1, 2, 5 and 10. The page follows those column captions. Original filename suffixes are retained in the manifest as source identifiers, not used to infer the displayed NFE.

| Paper column | Paper NFE | Original PCA filename | Original UMAP filename |
|---|---|---|---|
| 1 | 1 | `latent_pca_real_fit_1.jpg` | `latent_umap_real_fit_1.jpg` |
| 2 | 2 | `latent_pca_real_fit_5.jpg` | `latent_umap_real_fit_5.jpg` |
| 3 | 5 | `latent_pca_real_fit_10.jpg` | `latent_umap_real_fit_10.jpg` |
| 4 | 10 | `latent_pca_real_fit_50.jpg` | `latent_umap_real_fit_50.jpg` |

All eight images were inspected and contain no embedded NFE title that conflicts with the manuscript captions. Each manifest entry records the source image hash, copied asset hash, figure-source hash, projection, column and NFE. This establishes fidelity to the paper's presentation; historical checkpoints and paired-seed records remain unavailable.

### Grouped drift

The grouped-drift mapping preserves the manuscript panel order:

| Asset | Original filename | Meaning |
|---|---|---|
| `group-source-target.jpg` | `01_source_target.jpg` | Endpoint distributions: blue source `p_0`, red target `p_1` |
| `group-early.jpg` | `group_early_drift.jpg` | Group 1: `(t,r) = (0.05,0.30)` |
| `group-wide.jpg` | `group_wide_drift.jpg` | Group 2: `(t,r) = (0.20,0.80)` |
| `group-middle.jpg` | `group_middle_drift.jpg` | Group 3: `(t,r) = (0.35,0.65)` |
| `group-late.jpg` | `group_late_drift.jpg` | Group 4: `(t,r) = (0.70,0.95)` |

In the four time-pair panels, red denotes the target-time marginal, green denotes model output under the same model parameters, and dark arrows denote the grouped drift field. The grouping and colors follow the figure caption. These archived model visualizations have not been linked to recovered checkpoints.

MNIST's three class-grid / UMAP pairs in `img/img_MNIST.tex` similarly have paper column labels 1, 2 and 10 and source suffixes 1, 10 and 50. The inspected images have no embedded NFE title. These images are not included in the current page; the MNIST quantitative results are included below.

## Quantitative data

`results.json` contains three datasets with explicit protocols and release scope:

- **MNIST and FFHQ:** the complete eight-row comparison in the left table of `tab/tab_combine.tex`, label `tab:mnist_ffhq_comparison`. MNIST uses latent squared 2-Wasserstein EMD and classifier accuracy (stored in percent). FFHQ uses latent EMD and FID against decoded real latents using the same frozen ALAE decoder. That FID is not a raw-image-reference FFHQ metric.
- **ImageNet:** the four DFM L/2 rows in the active right table of `tab/tab_combine.tex`, label `tab:image256`: NFE 1/2/5/10, FID 1.52/1.45/1.34/1.31 and IS 259.4/268.3/279.8/287.4. This is the table included by `sec/05_experiment.tex`; the separately stored `tab/tab_imagenet.tex` is commented out and is not the data source. Evaluation uses 50,000 randomly generated images at 256 × 256. ImageNet code and checkpoints are outside this release.
- **Robotics:** two representative task-setting rows from the active table in `tab/tab_robotic.tex`, label `tab:robotics_control`. ToolHang State success rates are 0.41/0.77/0.81/0.86; PushT Visual rates are 0.86/0.87/0.89/0.90 at NFE 1/2/5/10. Values are stored as fractions and average the last 10 checkpoints. These are not the complete robotics table. Robotics code and checkpoints are outside this release.

The experiment descriptions come from `sec/05_experiment.tex`. All numbers are manuscript-reported results, not new measurements from the public package. JSON source hashes bind these values to the inspected manuscript files.

## Validation

From the repository root:

```bash
python scripts/check_site.py
```

This checks page references, figure source mappings and hashes, paper column labels, and result structure. It also compares the generated HTML cards, tables and initial FFHQ metrics against `results.json`. Interactive FFHQ metrics read the same generated table.

After changing result data, regenerate the static HTML blocks with:

```bash
python scripts/check_site.py --write-results
```

The page remains readable without JavaScript. CI checks the committed HTML and fails if a generated number is edited independently of its data source.

When the original manuscript checkout is available, use:

```bash
python scripts/check_site.py --manuscript-root /path/to/manuscript
```

This additionally checks source-file hashes, actual manuscript figure inclusion, time-pair labels, active table selection, and the benchmark values against the uncommented LaTeX tables. The script also detects an adjacent manuscript checkout automatically. In a standalone public clone it reports that released hashes, result structure and generated HTML were checked.

The HTML/CSS and interaction code were written for this project. No external project-page template or analytics script is bundled.
