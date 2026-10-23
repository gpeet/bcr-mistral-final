# Data layout

All code finds its files through three roots, set as environment variables (or in
`workflow/config.sh`). Each defaults to a folder inside the repository.

| Root | Default | Holds |
|---|---|---|
| `BCRM_DATA` | `data/` | Inputs: metadata, assembled repertoires, sequence tables, training corpora |
| `BCRM_MODELS` | `models/` | Released BCR-Mistral weights and the Hugging Face model config |
| `BCRM_OUT` | `results/` | Everything the pipeline writes: embeddings, classifiers, intermediates, figures |

Python code uses `bcrm.paths` (`data_path()`, `model_path()`, `out_path()`, `repo_path()`);
R code uses the same functions from `R/paths.R`.

"Source" says where a file comes from: **Zenodo** / **HF** (download), **stage** (made by
`workflow/run.sh <stage>`), or **seeds** (small files shipped in `seeds/` and copied into
`BCRM_OUT` automatically).

## BCRM_DATA

| Path | Contents | Source | Made by |
|---|---|---|---|
| `metadata/2024-SRARunTable-MetadataCleaning.xlsx` | Curated run metadata (disease, tissue, age, sex, treatment), version of 26 Nov 2025, the one the published figures and tables reproduce from | Zenodo | manual curation |
| `metadata/allOPIGdefinedUnpairedMetadata.csv` | OAS/OPIG study metadata | Zenodo | `dataset/02_extract_opig_metadata.ipynb` |
| `trust4/sra_runs.csv` | SRA runs to assemble (`run,layout,mode`) | stage | `sra-runlist` |
| `trust4/TRUST4results/*_airr.tsv` | TRUST4 assemblies, one per SRA run | Zenodo | `trust4` |
| `trust4/igblast_results/*_igblast.tsv` | IgBLAST re-annotation of the assemblies (not read by any figure) | not released | `igblast` |
| `opig/raw/*.csv.gz` | OPIG unpaired repertoires | OPIG | `opig-download` |
| `sequences/master-opig-dataset-sequence.csv` | Productive complete IGH from OPIG, top 25k per file | not released (19 GB) | `master-opig` |
| `sequences/master-trust4-dataset-sequence-unfiltered.csv` | All TRUST4 sequences (tab-separated); read by Fig. 1E / Table S1 | not released (23 GB) | `master-trust4` from the TRUST4 record |
| `sequences/master-trust4-dataset-unfiltered.csv` | June 2025 snapshot behind the Fig. 1E counts in the text: top 10,000 rows per file by consensus count, `sequence_alignment` column (11,563,296 rows); read by no code | not released | `master-trust4 --seq-col sequence_alignment --max-per-file 10000` (verified: same 11,563,296 rows; rows tied at the cutoff are chosen differently, so e.g. complete-VDJ rows come out at 2,906,406 vs 2,818,768) |
| `sequences/master-trust4-dataset-sequence.csv` | IGH-only extract of the above; superseded, read by no code | not released | `master-trust4 --chains IGH` |
| `sequences/preliminary-trust4-quality-sequences.csv` | Quality-filtered TRUST4 IGH sequences (consensus count > 2) with metadata (Fig. 1F–H) | Zenodo | `preliminary-trust4` |
| `sequences/all_high_quality_for_igblast.csv`, `all_hq_igblast_realigned.csv` | Dataset-build intermediates | Zenodo | `build-dataset` |
| `classifier/classifier_data_igblast-{train,test}-weights.csv` | Labelled sequence-classifier dataset, 2 outer folds | Zenodo | `build-dataset` |
| `classifier/treatment_sequences_{train,test}.csv` | Pre/post treatment sequences (full TRUST4 contigs; published Fig. 3B) | Zenodo | `build-dataset` |
| `classifier/treatment_sequences_vj_{train,test}.csv` | The same sequences IgBLAST V-J clipped and filtered like the classifier dataset (`sequence`), with the original `contig`, `v_sequence_start`, `j_sequence_end` and `contig_row` (row in the contig predictions) | stage | `treatment-vj-dataset` (`dataset/09_treatment_vj.R`) |
| `embedding-comparison/triplet_embedding_test_set_igblast.csv` | Triplet test set for Fig. 2C / S1A | Zenodo | `embeddings/compare_embeddings.Rmd` |
| `foundation/diverse_foundation_data.csv` | Diversity-selected pretraining corpus | Zenodo | `opig-shuffle` → `diversity` |
| `foundation/undiversified_foundation_data.csv` | Random-sample corpus (ablation): first N lines of the shuffled corpus | Zenodo | `undiversified` |
| `foundation/junction.csv` | CDR3 junction corpus | Zenodo | — |
| `foundation/diverse_foundation_data_protein.csv` | Translated corpus (protein model) | Zenodo | `foundation-models/protein_training_data.Rmd` |
| `foundation/classifier_data_igh_sequence-val.csv` | Validation sequences for model-size losses | Zenodo | — |
| `owens/MS Repertoire Fasta Files/` | Owens et al. antibody sequences; `Control-IgG/` holds the 2B4 isotype control | on request (`owens-fasta.tar.gz`; `fetch_data.sh owens`) | — |
| `comparator-output/<model>/{unflipped,flipped}/...` | Predictions of the k-mer / protein baseline classifiers used in the paper (Fig. 2E, 2F, S1B) | Zenodo | `comparators` with `BCRM_RETRAIN=1` |
| `owens/owens_sequences.csv` | Parsed Owens sequences (Fig. 3D) | on request (`fetch_data.sh owens`); without it the Owens stages and Fig. 3D are skipped | `owens-sequences` (`owens-antibodies/preprocess.Rmd`) |

## BCRM_MODELS

| Path | Model | Used by |
|---|---|---|
| `2025-HL-Foundation-1024x4096-HF/` | Hugging Face config for the 500M architecture | all 500M embedding scripts |
| `2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth` | the model named 500M (487 M parameters), 48 h checkpoint (converged): the one used in every analysis | all large-model embeddings, Fig. 2B |
| `trained/sequence-classifier/ensemble-output-*/{unflipped,flipped}/checkpoints/model_*.pt` | The paper's sequence classifiers: 10-model ensembles per outer fold (exclusion, baseline, junction-exclusion, junction-baseline) | `seqclf` (reused unless `BCRM_RETRAIN=1`) |
| `trained/sequence-classifier/waypoints{,-contrastive}-final/` | The paper's waypoint models and the patient feature tables computed from them (`patient_features*.csv`) | `waypoints` (reused unless `BCRM_RETRAIN=1`) |
| `2025-HL-Foundation-1024x4096-gradClip-junction.pth` | 500M architecture trained on junctions | junction embeddings |
| `2025-HL-Foundation-100M-gradClip.pth` | the model named 100M; 50 M parameters | model-size comparison |
| `2025-HL-Foundation-300M-gradClip.pth` | the model named 300M; 151 M parameters | model-size comparison, Fig. S1A |
| `2025-HL-Foundation-300M-gradClip-undiversified.pth` | 151 M, on the random-sample corpus | diversity ablation |
| `2026-Foundation-300M-gradClip-protein.pth` | the protein model named 300M; 151 M parameters | Fig. S1A |
| `2026-Foundation-100M-768x11.pth`, `2026-Foundation-300M-1024x20*.pth` | 104M / 304M replications (diverse, random-sample, protein). **Not released** (no published panel); trained locally by `run.sh pretrain 100M_768x11` etc. | supplementary size check beside Fig. 2B / S1A |
| `tokenizers/` | Copies of the two BPE tokenizers (also in the repository) | — |

All from Hugging Face Hub. Retraining (`workflow/run.sh pretrain <variant>`) writes to
`BCRM_OUT/foundation-models/checkpoints/`, never over these files. `eval-foundation` and `embed-compare`
read weights from `BCRM_MODELS`, so copy a retrained checkpoint there to evaluate it.

Tokenizers are small and live in the repository: `foundation-models/base_only_bpe_tokenizer/`
and `foundation-models/protein_bpe_tokenizer/`.

## BCRM_OUT

| Path | Contents | Source |
|---|---|---|
| `embeddings/` | Per-sequence embeddings (classifier, junction, treatment, comparison models); `treatment_embeddings_vj_{train,test}.csv` for the V-J clipped treatment sequences | `embed`, `embed-compare`, `embed treatment-vj` |
| `preprocessed/` | `.npy` caches of the embedding CSVs | created on first use |
| `sequence-classifier/ensemble-output-{exclusion,baseline,junction-exclusion,junction-baseline}/` | Sequence-classifier ensembles and predictions | `seqclf` |
| `sequence-classifier/comparator-output/` | k-mer / protein GLM and RF baselines | `comparators` |
| `sequence-classifier/hparam-search-*/best_config_*.json`, `waypoint*-hparam-search/best_config_*.json` | Tuned hyperparameters | seeds (or `tune`) |
| `sequence-classifier/waypoints-final/`, `waypoints-contrastive-final/` | Waypoint models and patient features | `waypoints` |
| `sequence-classifier/treatment_predictions_{exclusion,baseline}.csv` | Treatment-cohort predictions (full contigs; published Fig. 3B) | `treatment` |
| `sequence-classifier/treatment_predictions_vj_{exclusion,baseline}.csv` | Treatment-cohort predictions on the V-J clipped sequences | `treatment-vj` |
| `sequence-classifier/intermediates/` | Fig. 4A AUCs and SHAP feature ranking (`patient`), nearest training sequences per waypoint (`shap`), Fig. 4F tables | `patient`, `shap`, `figures` |
| `owens/` | Owens embeddings and predictions | `embed`, `owens` |
| `foundation-models/logs/training_loss-*.txt` | Pretraining loss logs | seeds (or `pretrain`) |
| `foundation-models/foundation_models_eval_losses.csv` | Validation losses by model | seeds (or `eval-foundation`) |
| `foundation/` | Rebuilt pretraining corpora (`shuffled_unselected_training_data.csv`, diverse, undiversified) | `opig-shuffle`, `diversity`, `undiversified` |
| `figures/` | Figure panels (SVG); `figures/batch-none/` holds the Fig. 4 panels without tissue batch correction (supplement) | `figures`, `patient` |
| `notebooks/` | Rendered R notebooks | `comparators`, `patient`, `figures` |
| `logs/` | Slurm logs | `workflow/submit_alpine.sh` |
