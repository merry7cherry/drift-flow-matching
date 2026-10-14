# Software verification

Checked on 2026-10-13 with Python 3.11.15, PyTorch 2.8.0, torchvision 0.23.0, NumPy 2.0.2 and Matplotlib 3.9.4 on macOS arm64. GitHub Actions runs the CPU checks on Ubuntu for each public commit.

## Training and evaluation

- **140 image tests passed:** grouped drift math, preprocessing, MNIST 16-dimensional defaults, all evaluation entrypoints, explicit PCA/UMAP selection, checkpoint/config compatibility, EMA/RNG resume, and synthetic train/sample/evaluate workflows.
- **7 trajectory tests passed:** DFM/FM/MeanFlow updates, checkpoint reload, matched source samples, singleton-group training and common plot bounds across methods/NFE.
- The image simplification was compared against public commit `b8f287c` on a fixed fixture: loss, 524 gradient entries, grouped drift, NFE 1/2/5 samples, old-checkpoint samples and one AdamW update were bitwise equal.
- DFM, FM and MeanFlow were compared against that same public version using seed 137, batch size 16 and four optimizer updates. Losses, parameters, NFE 1/2 states/times and old-checkpoint outputs were bitwise equal. Singleton-group drift is an intentional correction: it now yields conditional supervision instead of zero loss.
- Installed trajectory CLI training with singleton groups covered all four datasets. Reloading from outside the checkout reproduced all arrays in eight trace files exactly.

FFHQ evaluation defaults are shared by the general and dataset-specific entrypoints. Decoder noise is enabled by default, matching the supplied configuration; `--no-fid-decode-noise` disables decoder noise. DFM pretrained checkpoints are not distributed. See the [reproduction guide](README.md) for protocols and the relationship to the paper results.

## Project page

- **45 figure assets** checked against the original manuscript sources: 28 trajectory/Ground Truth panels, five grouped-drift panels, four FFHQ image grids and eight FFHQ PCA/UMAP panels.
- FFHQ coverage labels follow the active paper's column captions (1/2/5/10); the manifest preserves original filename mappings and hashes. Ground Truth uses the paper's unchanged first-column panels.
- HTML result cards and tables are generated from `docs/assets/results.json`. The FFHQ selector reads the generated table. Two regression tests cover visible-number drift and the paper's coverage mapping; source-data comparison also checks the active manuscript tables.
- Browser checks covered 32 dataset/method/NFE combinations, DFM step comparison, all four FFHQ budgets, all four training time pairs and Ground Truth switching. No browser errors were observed. Desktop and 390-pixel layouts had no page-level horizontal overflow; wide figures and tables scroll inside their own regions.
- Independent scientific and code reviews checked the method explanation, figure mapping, result order, configuration compatibility and evaluation defaults.

These checks establish bounded software behavior and fidelity to the paper's presentation. Full-budget benchmark training and CUDA execution were not rerun for this revision.
