# bcr-mistral

Code for *Language Model Embedding Classifiers Enable Identification of Multiple Sclerosis-Associated
BCRs and Repertoires* (Peet, Owens, Bennett, Krishnan, Macklin): decoder-only language models
(BCR-Mistral) trained on B cell receptor heavy-chain sequences, and the sequence- and patient-level
classifiers built on their embeddings to identify multiple sclerosis (MS)-associated BCRs.

Everything needed to regenerate the computational figure panels is here. Large inputs and model
weights are downloaded separately (see [Data](#data)).

## Layout

```
dataset/             SRA/TRUST4 + OPIG download and assembly, dataset building (00-09, *.Rmd)
foundation-models/   pretraining scripts, tokenizers, evaluation
embeddings/          embedding extraction (BCR-Mistral variants, ESM-2, Ab-RoBERTa) + comparison
sequence-classifier/ embedding classifiers (label exclusion / baseline) and waypoint models
owens-antibodies/    Owens et al. holdout antibodies: preprocessing, embeddings, prediction
analysis/            R notebooks behind the figures, SHAP nearest-sequence search
workflow/            run.sh (one entry point per stage), Slurm submission, site settings
envs/                pinned Python (ROCm) and R environments
seeds/               tuned hyperparameters and pretraining loss logs
bcrm/, R/paths.R     path helpers (see docs/DATA_LAYOUT.md)
scripts/             data download, release packaging, path check
docs/                data layout, verification, release READMEs
```

## Quick start

1. **Configure.** `cp workflow/config.example.sh workflow/config.sh` and set `BCRM_DATA`,
   `BCRM_MODELS`, `BCRM_OUT`. `docs/DATA_LAYOUT.md` lists every file under each root.
2. **Environments.** `workflow/run.sh env-python`, `env-r` (and `env-tools` to rebuild the dataset).
   GPU stages need PyTorch 1.13.0 (see [Hardware](#hardware)).
3. **Data.** `scripts/fetch_data.sh` downloads the processed data and weights and verifies their
   SHA-256 sums (see [Data](#data)).
4. **Run.** Stages in order:

   ```
   workflow/run.sh embed          # embeddings for classifier, treatment and Owens sets  (GPU)
   workflow/run.sh embed-compare  # embeddings for the model comparison (Fig. 2C, S1A)   (GPU)
   workflow/run.sh seqclf         # sequence classifiers, 4 variants                      (GPU)
   workflow/run.sh waypoints      # waypoint models and patient features                  (GPU)
   workflow/run.sh treatment      # pre/post treatment predictions                        (GPU)
   workflow/run.sh owens          # Owens holdout predictions (needs the on-request data) (GPU)
   workflow/run.sh comparators    # k-mer / protein baseline classifiers                  (R)
   workflow/run.sh patient        # patient classifier, SHAP                              (R)
   workflow/run.sh shap           # sequences nearest the top SHAP waypoints              (GPU)
   workflow/run.sh figures        # render all figure notebooks -> $BCRM_OUT/figures      (R)
   ```

   `workflow/run.sh list` shows every stage. On a Slurm cluster,
   `workflow/submit_alpine.sh downstream` submits the whole chain with dependencies (written for
   CU Boulder Alpine; change the partition lines for other clusters).

The stages reuse the paper's trained models by default — the classifier ensembles, waypoint models and
baseline predictions from the data release — so the figures come out as published. `BCRM_RETRAIN=1`
retrains and `workflow/run.sh tune` re-runs the hyperparameter searches; neither reproduces the
released models exactly, because the original training and searches were unseeded.
See `docs/VERIFICATION.md`.

## Models

| Model | Trained on | Used for |
|---|---|---|
| BCR-Mistral 487M (48 h checkpoint; files say `1024x4096`) | Diversity-selected OAS IGH nucleotide sequences | Full-sequence embeddings: sequence classifiers (2E, 2G/2H), waypoints and patient classifier (Fig. 4), treatment scores (3A-C), Owens predictions, model comparison (2B, 2C) |
| BCR-Mistral junction model (487M architecture) | CDR3 junction sequences only, 96-token context | Junction classifiers (2F) and the junction scores of the Owens antibodies (3D) |
| 50M, 151M (`256x1024`, `512x2048`) | Same corpus as the largest model | Model-size comparison (2B) |
| 151M protein | The diverse corpus translated to amino acids | Protein vs DNA comparison (S1A) |
| 151M random sample | Size-matched random IGH sample | Diversity-selection ablation (no published panel) |
| 104M, 304M, 304M protein, 304M random sample (`2026-Foundation-*`) — **weights not released** | Same corpora and recipe | Supplementary size check; losses are in `foundation_models_eval_losses.csv` |

Models are labelled by the parameter count of their configs. Each embedding type has its own
classifiers: `seqclf exclusion` / `baseline` use full-sequence embeddings, `seqclf
junction-exclusion` / `junction-baseline` the junction embeddings.

## Figures

Panels are written to `$BCRM_OUT/figures/`, named by manuscript panel (`fig2E_...`). Other plots the
notebooks make along the way have no `fig` prefix.

| Panel | File(s) | Notebook |
|---|---|---|
| 1A-D | `fig1A_*` ... `fig1D_*` | `analysis/sra_cohort_demographics.Rmd` |
| 1E, Table S1 | `fig1E_sequences_by_chain_and_tissue.csv`, `tableS1_sra_studies.csv` | `analysis/sra_sequence_counts.Rmd` |
| 1F, 1G | `fig1F_vgene_enrichment_csf_v_blood.svg`, `fig1G_vgene_enrichment_brain_v_blood.svg` | `analysis/v_gene_enrichment.Rmd` |
| 1H | `fig1H_clone_rate_enrichment.svg` | `analysis/public_clones.Rmd` |
| 1I | `fig1I_cdr3_length_density.svg`, `fig1I_cdr3_length_tests.csv` | `analysis/sra_cdr3_assembly_lengths.Rmd` |
| 2B | `fig2B_validation_loss_by_model_size.svg` | `foundation-models/model-size-comparison.Rmd` |
| 2C, S1A | `fig2C_embedding_comparison_{cosine,triplet}.svg`, `figS1A_*` | `embeddings/compare_embeddings.Rmd` |
| 2E, S1B | `fig2E_precision_fold_curve.svg`, `figS1B_nmo_precision_fold_curve.svg` | `analysis/evaluate_classifier.Rmd` |
| 2F | `fig2F_precision_fold_curve_junction.svg` | `analysis/evaluate_classifier_junction_embedding.Rmd` |
| 2G, 2H | `fig2G_vgene_enrichment_forest.svg`, `fig2H_vgene_frequency_top15.svg` | `analysis/v_gene_enrichment_classifier.Rmd` |
| 3A, 3C | `fig3A_patient_model_comparison.svg`, `fig3C_patient_disease_class.svg` | `analysis/patient_analysis.Rmd` |
| 3B, S2 | `fig3B_treatment_trajectories_*.svg` | `analysis/treatment_analysis.Rmd` |
| 3D | `fig3D_owens_prediction.svg` (needs the on-request Owens sequences; skipped otherwise) | `owens-antibodies/analysis.Rmd` |
| 4A-E | `fig4A_feature_set_roc.svg` ... `fig4E_max_sequences_auc.svg` | `analysis/patient_classifier.Rmd` |
| 4F | `fig4F_shap_waypoint_vgene_heatmap.svg` | `analysis/shap_waypoint_sequences.Rmd` |
| Supplement: 3B on V-J clipped sequences | `fig3B_*_vj.svg`, `figS_treatment_contig_vs_vj.svg` | stages `treatment-vj-dataset`, `embed treatment-vj`, `treatment-vj`, `treatment-vj-figures` |
| Supplement: 4A-E without tissue batch correction | `batch-none/fig4*_*.svg` | `patient` stage (`BCRM_BATCH_CORRECTION=none`) |

Two notebooks feed panels without producing one: `analysis/baseline_models.Rmd` trains the 2E/2F
comparators (stage `comparators`, only with `BCRM_RETRAIN=1`), and
`foundation-models/protein_training_data.Rmd` translates the protein model's corpus.

Schematics (Fig. 1 workflow, 2A, 2D) and the flow-cytometry panel (3E) are not computational and are
not reproduced here.

## Data

| What | Where | `scripts/fetch_data.sh` |
|---|---|---|
| Processed data: metadata, labelled classifier sets, pretraining corpora, triplet test set, Fig. 1F-H table, baseline predictions (~6 GB unpacked) | Zenodo [10.5281/zenodo.23605219](https://doi.org/10.5281/zenodo.23605219) | `data` |
| TRUST4 assemblies, one AIRR file per run (~47 GB unpacked): needed for Fig. 1I and 1E / Table S1, and to rebuild the dataset | the same Zenodo record | `trust4` |
| BCR-Mistral weights, transformers config, tokenizers, trained classifiers and waypoint models | Hugging Face [grpeet/bcr-mistral](https://huggingface.co/grpeet/bcr-mistral) (gated: accept the terms on the model page) | `models` |
| Owens et al. CSF plasmablast antibody sequences (Fig. 3D): not public | On request from Graham Peet (graham.peet@cuanschutz.edu); set `OWENS_BUNDLE` to the folder | `owens` |
| Raw repertoires | NCBI SRA (accessions in the metadata table) and OPIG/OAS | rebuild stages |

The Zenodo link above is version 1.0.1, the version these files were released as;
[10.5281/zenodo.20432912](https://doi.org/10.5281/zenodo.20432912) resolves to the latest version
instead. `scripts/fetch_data.sh` pins the version, so it fetches the same files every time;
`ZENODO_DATA_RECORD` overrides it.

Without the TRUST4 assemblies, `workflow/run.sh figures` skips Fig. 1E / Table S1 and 1I and says so.
The master sequence tables are not distributed; `master-trust4` and `master-opig` rebuild them.

## Rebuilding the dataset and models

The heavy upstream stages are runnable but not needed to regenerate the figures:

```
workflow/run.sh sra-runlist               # SRA runs to assemble, with their TRUST4 mode
workflow/run.sh trust4 <index>            # download one run + TRUST4 (array job)
workflow/run.sh igblast <batch>           # IgBLAST re-annotation (array job)
workflow/run.sh master-trust4 [--chains IGH|all]
workflow/run.sh opig-download             # OPIG/OAS unpaired repertoires
workflow/run.sh master-opig               # productive complete IGH, top 25k per file
workflow/run.sh opig-shuffle              # IGH corpus in random order
workflow/run.sh diversity                 # diversity-selected pretraining corpus
workflow/run.sh undiversified             # size-matched random-sample corpus (ablation)
workflow/run.sh preliminary-trust4        # quality-filtered IGH table behind Fig. 1F-H
workflow/run.sh build-dataset             # labelled classifier dataset
workflow/run.sh treatment-vj-dataset      # treatment sequences V-J clipped (Fig. 3B supplement)
workflow/run.sh owens-sequences           # Owens antibody FASTAs -> annotated sequence table
workflow/run.sh pretrain <model>          # see run.sh list for the model names
workflow/run.sh eval-foundation
```

A few runs needed a non-standard TRUST4 mode; `dataset/trust4_run_modes.csv` records which, and
`workflow/submit_alpine.sh trust4 standard|qc|10x` submits each mode with suitable resources. The
pretraining corpora were drawn from an unseeded shuffle, so rebuilding them gives a different but
equivalent selection; the released corpora are the ones the models were trained on.

TRUST4 is installed separately (https://github.com/liulab-dfci/TRUST4; set `TRUST4_HOME`). IgBLAST and
its IMGT databases come from the `igblastr` R package (set `IGBLASTR_CACHE`).

For smoke tests, `MAX_SPOTS` limits reads per run, `BCRM_MAX_STEPS` and `BCRM_MAX_EPOCHS` cut training
short, and `BCRM_TUNE_TRIALS` sets the number of search trials. Retrained checkpoints go to
`$BCRM_OUT/foundation-models/checkpoints/` and never overwrite the released weights.
`scripts/check_no_hardcoded_paths.sh` fails if any file refers to a local cluster path.

## Not used for the paper

Kept for reference; no stage runs them and no panel depends on them:

- `embeddings/extract_embeddings_{meanpool,attnpool,junction_meanpool,treatment_meanpool}.py`,
  `extract_mistral_attention_pooled.py`, `extract_mistral_last_token.py`: pooling-method ablations
  (the paper uses last-token embeddings from `extract_embeddings*.py`).
- `analysis/evaluate_classifier_precision.Rmd`: a precision-loss classifier variant.
- `foundation-models/CustomTokenizer.ipynb`: how the BPE tokenizers were built.

## Hardware

GPU stages were run on AMD Instinct MI100 GPUs with CU Boulder Research Computing's ROCm 5.2 build of
PyTorch 1.13.0 (`workflow/site/alpine.sh`). `envs/setup_python_env.sh` installs the pinned packages on
top of an existing PyTorch; `BCRM_INSTALL_TORCH=1` installs the ROCm wheel too. A CUDA build of
PyTorch 1.13.0 should also work, but the results were not produced with one.

## License

Copyright 2026 The Regents of the University of Colorado. **All rights reserved**; see `LICENSE`.
No copyright or patent licence is granted at this time. You may view, download and run this code and
the model weights to examine or verify the results in the paper; any other use needs written
permission, and a licence for wider use is expected to follow. For permissions, contact Graham Peet
(graham.peet@cuanschutz.edu).

The processed data on Zenodo are released separately under **CC BY 4.0**. The Owens et al. antibody
sequences behind Fig. 3D are not public and are shared on request.
