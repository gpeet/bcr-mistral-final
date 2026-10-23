#!/bin/bash
set -euo pipefail

# Run one stage of the bcr-mistral pipeline in the current shell / allocation.
#
# Usage:
#   workflow/run.sh <stage> [args]
#   workflow/run.sh list
#
# Configuration comes from workflow/config.sh (copy workflow/config.example.sh).
# Each stage activates its environment through the site file named by BCRM_SITE
# (workflow/site/<site>.sh), which defines bcrm_env_gpu, bcrm_env_r and bcrm_env_tools.
# On a Slurm cluster, workflow/submit_alpine.sh shows how to chain the stages as jobs.

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
[ -f workflow/config.sh ] && source workflow/config.sh
export BCRM_DATA=${BCRM_DATA:-$REPO_ROOT/data}
export BCRM_MODELS=${BCRM_MODELS:-$REPO_ROOT/models}
export BCRM_OUT=${BCRM_OUT:-$REPO_ROOT/results}
source "workflow/site/${BCRM_SITE:-generic}.sh"

render() {  # render R Markdown notebooks (HTML to $BCRM_OUT/notebooks); report every failure at the end
    local failed=()
    for rmd in "$@"; do
        echo "== render $rmd"
        Rscript -e "rmarkdown::render('$rmd', output_dir = '$BCRM_OUT/notebooks', quiet = TRUE)" || failed+=("$rmd")
    done
    [ ${#failed[@]} -eq 0 ] || { echo "== FAILED: ${failed[*]}"; return 1; }
}

py() { echo "== python $*"; python -u "$@"; }

# Second Fig. 3B variant: the treatment sequences V-J clipped like the classifier training data
# (treatment-vj-dataset, embed treatment-vj, treatment-vj). Renders the Fig. 3B notebook on them
# (figures with a _vj suffix, notebook treatment_analysis_vj.html) and the contig vs V-J comparison.
treatment_vj_figures() {
    echo "== render analysis/treatment_analysis.Rmd (sequences = vj)"
    Rscript -e "rmarkdown::render('analysis/treatment_analysis.Rmd', params = list(sequences = 'vj'),
        output_dir = '$BCRM_OUT/notebooks', output_file = 'treatment_analysis_vj.html', quiet = TRUE)"
    render analysis/treatment_vj_comparison.Rmd
}

# Copy the shipped seeds/ (tuned hyperparameters, pretraining logs) into BCRM_OUT,
# never overwriting files a stage has already produced.
seed_outputs() {
    (cd seeds && find . -type f) | while read -r f; do
        [ -e "$BCRM_OUT/$f" ] || { mkdir -p "$(dirname "$BCRM_OUT/$f")"; cp "seeds/$f" "$BCRM_OUT/$f"; }
    done
}

# Released trained models (the paper's sequence classifiers and waypoints) are copied into BCRM_OUT,
# so the stages reuse them and reproduce the figures exactly. BCRM_RETRAIN=1 retrains instead.
use_released() {  # use_released <path under sequence-classifier/>
    [ "${BCRM_RETRAIN:-0}" = 1 ] && return 0
    local src="$BCRM_MODELS/trained/sequence-classifier/$1"
    [ -e "$src" ] || { echo "no released model at $src: training instead"; return 0; }
    mkdir -p "$BCRM_OUT/sequence-classifier"
    rm -rf "$BCRM_OUT/sequence-classifier/$1"   # replace anything from an earlier retrain
    cp -r "$src" "$BCRM_OUT/sequence-classifier/"
}

# summarize <script> <features csv under sequence-classifier/>: patient features from the waypoints.
# They are computed on the GPU, where float sums are not bit-reproducible, and the patient classifier's
# hyperparameter search amplifies even 1e-6 differences; so the released features are used unless
# BCRM_RETRAIN=1 (or they are missing).
summarize() {
    if [ "${BCRM_RETRAIN:-0}" != 1 ] && [ -s "$BCRM_OUT/sequence-classifier/$2" ]; then
        echo "== using released $2"
    else py "$1"; fi
}

# The Owens antibody sequences (Fig. 3D) are not public; they are shared on request (README "Data")
owens_data() {
    [ -s "$BCRM_DATA/owens/owens_sequences.csv" ] && return 0
    echo "== skip Owens antibodies: $BCRM_DATA/owens/owens_sequences.csv not present (shared on request)"; return 1
}

conda_env() {  # conda_env <name> <yml>: create the environment, or bring an existing one up to the yml
    bcrm_conda
    if conda env list | grep -q "^$1 "; then conda env update -n "$1" -f "$2"
    else conda env create -y -n "$1" -f "$2"; fi
}

TOKENIZER=foundation-models/base_only_bpe_tokenizer/tokenizer.json
MODEL_HF=$BCRM_MODELS/2025-HL-Foundation-1024x4096-HF
TRIPLETS=$BCRM_DATA/embedding-comparison/triplet_embedding_test_set_igblast.csv

stage=${1:-list}; shift || true
[ "$stage" = list ] || seed_outputs
case "$stage" in

# ── Environments ─────────────────────────────────────────────────────────────
env-python)  bcrm_env_gpu_base; envs/setup_python_env.sh ;;
env-r)       conda_env "$BCRM_R_ENV" envs/environment-r.yml; bcrm_env_r; envs/setup_r_env.sh ;;
env-tools)   conda_env "$BCRM_TOOLS_ENV" envs/environment-tools.yml ;;

# ── Dataset assembly (heavy; see README "Rebuilding the dataset") ─────────────
sra-runlist) bcrm_env_r; Rscript dataset/00_make_run_list.R "$BCRM_DATA/trust4/sra_runs.csv" ;;
trust4)      bcrm_env_tools; dataset/03_trust4_sra.sh "$BCRM_DATA/trust4/sra_runs.csv" "$@" ;;
igblast)     dataset/igblast_trust4.sh "$@" ;;
opig-download) bcrm_env_tools; dataset/01_download_opig.sh dataset/allOPIGsubjectDefinedUnpaired.txt "$BCRM_DATA/opig/raw" ;;
opig-shuffle) bcrm_env_r; Rscript dataset/04_shuffle_opig.R "$BCRM_DATA/sequences/master-opig-dataset-sequence.csv" \
                 "$BCRM_OUT/foundation/shuffled_unselected_training_data.csv" ;;
diversity)   bcrm_env_gpu
             in=$BCRM_OUT/foundation/shuffled_unselected_training_data.csv
             py dataset/05_select_diverse_sequences.py --input "$in" --n-sequences "$(wc -l < "$in")" \
                 --output "$BCRM_OUT/foundation/diverse_foundation_data.csv" "$@" ;;
undiversified) # random-sample ablation corpus: as many sequences as the diverse corpus, in shuffled order
             head -n "$(wc -l < "$BCRM_OUT/foundation/diverse_foundation_data.csv")" \
                 "$BCRM_OUT/foundation/shuffled_unselected_training_data.csv" > "$BCRM_OUT/foundation/undiversified_foundation_data.csv" ;;
master-opig)  bcrm_env_tools; dataset/06_master_opig.sh "$BCRM_DATA/opig/raw" "$BCRM_DATA/sequences/master-opig-dataset-sequence.csv" ;;
master-trust4) bcrm_env_tools; dataset/07_master_trust4.sh "$BCRM_DATA/trust4/TRUST4results" "$BCRM_DATA/sequences/master-trust4-dataset-sequence-unfiltered.csv" "$@" ;;
preliminary-trust4) bcrm_env_r; Rscript dataset/08_preliminary_trust4.R "$BCRM_DATA/trust4/TRUST4results" \
                 "$BCRM_DATA/sequences/master-trust4-dataset-sequence-unfiltered.csv" \
                 "$BCRM_DATA/metadata/2024-SRARunTable-MetadataCleaning.xlsx" \
                 "$BCRM_DATA/sequences/preliminary-trust4-quality-sequences.csv" ;;
build-dataset) bcrm_env_r; render dataset/build_dataset.Rmd dataset/build_treatment_dataset.Rmd ;;
treatment-vj-dataset) bcrm_env_r   # IgBLAST + V-J clip of the treatment sequences (second Fig. 3B variant)
             Rscript dataset/09_treatment_vj.R \
                 "$BCRM_DATA/classifier/treatment_sequences_train.csv" "$BCRM_DATA/classifier/treatment_sequences_test.csv" \
                 "$BCRM_DATA/classifier/treatment_sequences_vj_train.csv" "$BCRM_DATA/classifier/treatment_sequences_vj_test.csv" ;;
owens-sequences) bcrm_env_r; render owens-antibodies/preprocess.Rmd ;;   # Owens antibody FASTAs -> owens/owens_sequences.csv

# ── Foundation models (heavy) ────────────────────────────────────────────────
pretrain)    bcrm_env_gpu
             case "${1:?pretrain needs a variant: 100M 300M 300M_protein 300M_undiversified 500M 500M_junction (+ 100M_768x11 300M_1024x20 300M_protein_1024x20 300M_undiversified_1024x20)}" in
                 100M|300M|300M_protein|300M_undiversified|500M|500M_junction) py "foundation-models/train_$1.py" ;;
                 100M_768x11|300M_1024x20|300M_protein_1024x20|300M_undiversified_1024x20) py "foundation-models/train_$1.py" ;;  # correctly sized
                 *) echo "unknown variant $1"; exit 1 ;;
             esac ;;
eval-foundation) bcrm_env_gpu
             py foundation-models/evaluate.py --eval-csv "$BCRM_DATA/foundation/classifier_data_igh_sequence-val.csv" \
                 --output-csv "$BCRM_OUT/foundation-models/foundation_models_eval_losses.csv" \
                 --tokenizer "$TOKENIZER" --model-dir "$BCRM_MODELS" \
                 --indices-from seeds/foundation-models/foundation_models_eval_losses.csv ;;

# ── Embeddings (GPU) ─────────────────────────────────────────────────────────
embed)       bcrm_env_gpu   # optional part: main | main-train | main-test | junction | treatment | treatment-vj | owens
             part=${1:-all}
             [[ $part =~ ^(all|main)$ ]]      && py embeddings/extract_embeddings.py           # 500M, classifier train/test
             [[ $part == main-train ]]        && py embeddings/extract_embeddings.py train
             [[ $part == main-test ]]         && py embeddings/extract_embeddings.py test
             [[ $part =~ ^(all|junction)$ ]]  && py embeddings/extract_embeddings_junction.py  # junction model, classifier train/test
             [[ $part =~ ^(all|treatment)$ ]] && py embeddings/extract_embeddings_treatment.py # 500M, pre/post treatment sequences
             [[ $part == treatment-vj ]]      && py embeddings/extract_embeddings_treatment.py --variant vj  # V-J clipped (not in all)
             if [[ $part =~ ^(all|owens)$ ]]; then
                 if owens_data; then
                     py owens-antibodies/extract_embeddings.py
                     py owens-antibodies/extract_junction_embeddings.py
                 fi
             fi ;;
embed-compare) bcrm_env_gpu
             # Fig 2C / S1A. The 500M model is the 48 h checkpoint, as in every other analysis.
             py embeddings/extract_mistral_pooled.py --input "$TRIPLETS" \
                 --out-mean "$BCRM_OUT/embeddings/mistral-meanpooled-embeddings.csv" \
                 --out-max "$BCRM_OUT/embeddings/mistral-maxpooled-embeddings.csv" \
                 --out-weighted "$BCRM_OUT/embeddings/mistral-weightedpooled-embeddings.csv" \
                 --model-hf "$MODEL_HF" --weights "$BCRM_MODELS/2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth" \
                 --tokenizer "$TOKENIZER" --sequence-col sequence
             py embeddings/extract_mistral_300m_pooled.py
             py embeddings/extract_mistral_300m_undiversified_pooled.py
             py embeddings/extract_mistral_300m_protein_pooled.py
             for v in diverse undiversified protein; do   # correctly sized 304M models, if trained
                 case $v in diverse) w=2026-Foundation-300M-1024x20.pth ;; *) w=2026-Foundation-300M-1024x20-$v.pth ;; esac
                 [ -e "$BCRM_MODELS/$w" ] && py embeddings/extract_mistral_1024x20_pooled.py $v
             done
             py embeddings/extract_esm2.py --input "$TRIPLETS" --output "$BCRM_OUT/embeddings/esm2-meanpooled-embeddings.csv"
             py embeddings/extract_abroberta.py --input "$TRIPLETS" --output "$BCRM_OUT/embeddings/abroberta-meanpooled-embeddings.csv" ;;

# ── Classifiers (GPU) ────────────────────────────────────────────────────────
seqclf)      bcrm_env_gpu   # optional part: exclusion | baseline | junction-exclusion | junction-baseline
             part=${1:-all}
             for v in exclusion baseline junction-exclusion junction-baseline; do
                 [[ $part =~ ^(all|$v)$ ]] && use_released "ensemble-output-$v"
             done
             [[ $part =~ ^(all|exclusion)$ ]]          && py sequence-classifier/train_embedding_classifier.py
             [[ $part =~ ^(all|baseline)$ ]]           && py sequence-classifier/train_embedding_classifier_baseline.py
             [[ $part =~ ^(all|junction-exclusion)$ ]] && py sequence-classifier/train_embedding_classifier_junction.py
             [[ $part =~ ^(all|junction-baseline)$ ]]  && py sequence-classifier/train_embedding_classifier_junction_baseline.py
             true ;;
waypoints)   bcrm_env_gpu
             use_released waypoints-final; use_released waypoints-contrastive-final
             py sequence-classifier/waypoints/train_waypoints.py
             py sequence-classifier/waypoints/train_waypoints_contrastive.py
             summarize sequence-classifier/waypoints/summarize_patients.py waypoints-final/patient_features.csv
             summarize sequence-classifier/waypoints/summarize_patients_contrastive.py \
                 waypoints-contrastive-final/patient_features_contrastive.csv ;;
treatment)   bcrm_env_gpu
             py sequence-classifier/predict_from_embeddings.py
             py sequence-classifier/predict_from_embeddings_baseline.py ;;
treatment-vj) bcrm_env_gpu   # second Fig. 3B variant (after treatment-vj-dataset and embed treatment-vj)
             py sequence-classifier/predict_from_embeddings.py --variant vj
             py sequence-classifier/predict_from_embeddings_baseline.py --variant vj ;;
owens)       bcrm_env_gpu; if owens_data; then py owens-antibodies/predict.py; fi ;;
tune)        bcrm_env_gpu   # optional: re-run the Optuna searches (unseeded; 200 trials each). The new best_config
             # files replace the shipped ones in BCRM_OUT and are what BCRM_RETRAIN=1 then trains with
             py sequence-classifier/tune_embedding_classifier.py
             py sequence-classifier/tune_embedding_classifier_baseline.py
             py sequence-classifier/waypoints/tune_waypoints.py
             py sequence-classifier/waypoints/tune_waypoints_contrastive.py ;;

# ── Analysis and figures (R, CPU) ────────────────────────────────────────────
comparators) # released k-mer/protein baseline predictions (Zenodo) unless BCRM_RETRAIN=1
             if [ "${BCRM_RETRAIN:-0}" != 1 ] && [ -d "$BCRM_DATA/comparator-output" ]; then
                 mkdir -p "$BCRM_OUT/sequence-classifier"
                 rm -rf "$BCRM_OUT/sequence-classifier/comparator-output"
                 cp -r "$BCRM_DATA/comparator-output" "$BCRM_OUT/sequence-classifier/"
             else bcrm_env_r; render analysis/baseline_models.Rmd; fi ;;
patient)     bcrm_env_r; render analysis/patient_classifier.Rmd
             # Supplementary figure: the same analysis without the tissue-source batch correction
             # (figures/batch-none/, intermediates/batch-none/)
             echo "== render analysis/patient_classifier.Rmd (BCRM_BATCH_CORRECTION=none)"
             BCRM_BATCH_CORRECTION=none Rscript -e "rmarkdown::render('analysis/patient_classifier.Rmd',
                 output_dir = '$BCRM_OUT/notebooks', output_file = 'patient_classifier_batch-none.html', quiet = TRUE)" ;;
shap)        bcrm_env_gpu; py analysis/shap_nearest_sequences.py ;;
figures)     bcrm_env_r   # figures [notebook.Rmd ...]: default renders them all
             if [ $# -gt 0 ]; then render "$@"; else
             nbs=(analysis/sra_cohort_demographics.Rmd analysis/sra_sequence_counts.Rmd analysis/v_gene_enrichment.Rmd
                  analysis/public_clones.Rmd analysis/sra_cdr3_assembly_lengths.Rmd
                  foundation-models/model-size-comparison.Rmd embeddings/compare_embeddings.Rmd
                  analysis/evaluate_classifier.Rmd analysis/evaluate_classifier_junction_embedding.Rmd
                  analysis/v_gene_enrichment_classifier.Rmd analysis/patient_analysis.Rmd
                  analysis/treatment_analysis.Rmd owens-antibodies/analysis.Rmd analysis/shap_waypoint_sequences.Rmd)
             # Fig. 3D needs the Owens antibody sequences, which are shared on request only
             if [ ! -s "$BCRM_OUT/owens/owens_predictions_per_model.csv" ]; then
                 echo "== skip Fig. 3D: no Owens predictions (the Owens sequences are shared on request; see README)"
                 nbs=("${nbs[@]/owens-antibodies\/analysis.Rmd}"); fi
             # Fig. 1E/Table S1 and 1I read the TRUST4 assemblies (fetch_data.sh trust4; 1E also run.sh master-trust4)
             if [ ! -s "$BCRM_DATA/sequences/master-trust4-dataset-sequence-unfiltered.csv" ]; then
                 echo "== skip Fig. 1E / Table S1: needs fetch_data.sh trust4, then workflow/run.sh master-trust4"
                 nbs=("${nbs[@]/analysis\/sra_sequence_counts.Rmd}"); fi
             if ! compgen -G "$BCRM_DATA/trust4/TRUST4results/*airr.tsv" >/dev/null; then
                 echo "== skip Fig. 1I: needs fetch_data.sh trust4"
                 nbs=("${nbs[@]/analysis\/sra_cdr3_assembly_lengths.Rmd}"); fi
             render ${nbs[@]}
             if [ -s "$BCRM_OUT/sequence-classifier/treatment_predictions_vj_exclusion.csv" ]; then treatment_vj_figures
             else echo "== skip the V-J variant of Fig. 3B: run treatment-vj-dataset, embed treatment-vj, treatment-vj"; fi
             fi ;;
treatment-vj-figures) bcrm_env_r; treatment_vj_figures ;;

list)        sed -n 's/^\([a-z0-9-]*\))  *.*/\1/p' "$0" | grep -v '^list$' ;;
*)           echo "unknown stage: $stage (try: $0 list)"; exit 1 ;;
esac
