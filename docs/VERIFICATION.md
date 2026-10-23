# Verification

What was tested when this code was prepared for release, and how to reproduce it. The panel-by-panel
map of figures to notebooks is in the [README](../README.md#figures).

## Environments

The environments were built from scratch twice, once in the working directory and once in a clean room
from a fresh clone, and the figures were regenerated in each:

- Python from `envs/requirements-rocm.txt` on PyTorch 1.13.0 + ROCm 5.2 (transformers 4.46.1,
  optuna 4.7.0).
- R 4.4.1 from `renv.lock` (229 packages).
- Dataset tools from `envs/environment-tools.yml`.

`scripts/check_no_hardcoded_paths.sh` fails if any file refers to a machine-specific path; it passes on
the released tree.

## Reproducing the figures

With the released data and model weights in place (`scripts/fetch_data.sh`), running the stages in the
order the [README](../README.md#quick-start) lists regenerates every computational panel of Figures 1-4
and S1-S2 into `$BCRM_OUT/figures/`. `workflow/run.sh figures` renders the figure notebooks; Fig. 4A-E
come from the `patient` stage and 4F from `shap`, so those run first.

Fig. 3D needs the Owens antibody sequences, which are not part of the public release (see the
[README](../README.md#data)); the figures stage skips that panel when they are absent. The Figure 1
workflow diagram and panels 2A, 2D and 3E are schematics and flow cytometry, not computational.

Downstream results are deterministic on CPU. GPU arithmetic is not bit-reproducible, so embeddings and
patient features differ by around 1e-6 between runs; the released feature tables are reused by default
so that the figures regenerate exactly. `BCRM_RETRAIN=1` retrains instead, which gives similar but not
identical models — see [Rebuilding the dataset and models](../README.md#rebuilding-the-dataset-and-models).

## Heavy stages

These stages are expensive to run in full, so each was tested end to end on a small input to confirm it
runs and writes the expected output. Rerunning them in full is not needed to reproduce the figures.

| Stage | Test input |
|---|---|
| SRA download and TRUST4 assembly, all four modes | 200,000 spots per run: ERR4690319 (paired), ERR3504638 (single-end), ERR4690321 (`qc`), SRR12483498 (`10x` barcode/UMI) |
| IgBLAST re-annotation | the four assemblies above; 114-column AIRR tables |
| OPIG shuffle, diversity selection, undiversified corpus | 20,000 sequences |
| Foundation-model pretraining, all six models | 50-59 steps each; the checkpoints load and `foundation-models/evaluate.py` runs on them |

The IgBLAST re-annotation behind the released sequence table was also checked against it directly:
re-running IgBLAST over 20,000 released sequences reproduced every annotated column exactly.

## Release packaging

`scripts/make_release.sh <stage_dir>` builds the Zenodo and Hugging Face bundles, each with a README and
`SHA256SUMS`, and `scripts/fetch_data.sh` unpacks them back into the expected layout. The round trip was
tested: packaged, re-fetched into an empty data directory, and the figures regenerated from the result.
