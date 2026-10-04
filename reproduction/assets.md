# Project-page figure sources

The project page uses archived manuscript figures. No model outputs were created or improved by an image-generation tool.

- `docs/assets/ffhq-nfe-{1,2,5,10}.jpg`: raster exports of the corresponding manuscript appendix PDFs, `img/optimized/result_FFHQ/ffhq_fake_steps_grid_step_{N}.pdf`. The captions and embedded figure titles agree with these NFE values. Each export uses the complete PDF page at a maximum dimension of 1800 pixels. Selection and seed pairing are not recoverable, so columns must not be described as a paired-seed comparison.
- `docs/assets/{dataset}-{method}.jpg`: unchanged optimized figures referenced in the manuscript's `img/img_viz.tex`. Dataset names are moon, checkerboard_grid, letter_f, and letter_m. Display labels follow that manuscript figure: Flow Matching 50, MeanFlow 1/20, Drift Model 1, and DFM 1/20.

The source-to-asset manifest and checksums are in [provenance.json](../docs/assets/provenance.json). These archived assets have not been associated with recovered model checkpoints.

MNIST class-grid and FFHQ latent-projection figures are omitted from the step selector because their stored filenames and manuscript NFE labels disagree. We do not resolve that disagreement by relabeling the figures.

The HTML/CSS and interaction code were written for this project. No external project-page template or analytics script is bundled.
