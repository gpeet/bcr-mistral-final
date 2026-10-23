#!/bin/bash
set -euo pipefail

# Package BCRM_DATA and BCRM_MODELS into upload-ready bundles for Zenodo and the Hugging Face Hub.
# Nothing is uploaded; review the staging folder, then upload by hand.
#
# Usage:
#   scripts/make_release.sh <stage_dir>
#
# Writes:
#   <stage_dir>/zenodo-data/     processed data (classifier sets, corpora, metadata, ...) and the
#                                TRUST4 AIRR assemblies -- one Zenodo record, released CC BY 4.0
#   <stage_dir>/huggingface/     model weights + transformers config + tokenizers + trained models
#
# Zenodo records are flat, so the TRUST4 assemblies are tarred and every basename must be unique.
# The weights go to Hugging Face rather than Zenodo: they are not released under the data record's
# CC BY 4.0 terms, and a Zenodo record carries only one licence.
# each with a README (from docs/release/) and SHA256SUMS. scripts/fetch_data.sh unpacks the Zenodo files back into the BCRM_DATA layout.
#
# Not packaged (rebuildable, see README "Rebuilding the dataset"): raw OPIG downloads, the
# master OPIG/TRUST4 sequence tables. Their builders are verified against the originals.

STAGE=${1:?usage: $0 <stage_dir>}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BCRM_DATA=${BCRM_DATA:-$REPO_ROOT/data}
BCRM_MODELS=${BCRM_MODELS:-$REPO_ROOT/models}
GZ=$(command -v pigz || echo gzip)

# fresh <dest> <source...>: dest exists and nothing in the sources changed since it was written
fresh() { local d=$1; shift; [ -s "$d" ] && [ -z "$(find "$@" -newer "$d" -print -quit)" ]; }

TAR_OPTS=(--owner=0 --group=0 --numeric-owner)   # no cluster user names in the archives

gz_into() {  # gz_into <dest_dir> <file relative to BCRM_DATA>: dest/<rel>.gz, keeping the folder structure
    mkdir -p "$1/$(dirname "$2")"
    fresh "$1/$2.gz" "$BCRM_DATA/$2" || { "$GZ" -c "$BCRM_DATA/$2" > "$1/$2.gz.part" && mv "$1/$2.gz.part" "$1/$2.gz"; }
}

sums() { (cd "$1" && find . -type f ! -name SHA256SUMS ! -name "*.part" -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS); }

# Zenodo records are flat: upload every file under zenodo-*/ by basename (fetch_data.sh puts
# them back into folders from SHA256SUMS), so basenames must be unique.
flat_ok() { local d; d=$(find "$1" -type f -printf '%f\n' | sort | uniq -d); [ -z "$d" ] || { echo "duplicate basenames in $1: $d"; exit 1; }; }

# Sequence identifiers that must never appear in a released bundle. The release fails if one does.
# `need` keeps the scan honest: a missing archive is a failure, not a silent pass, so this cannot rot
# into a no-op that reports clean because it found nothing to look at.
EXCLUDED='ic05'
need() { [ -s "$1" ] || { echo "exclusion check: expected $1 to exist"; exit 1; }; }
no_excluded() {
    local hit n
    hit=$(find "$STAGE" -iname "*$EXCLUDED*" -print)
    [ -z "$hit" ] || { echo "exclusion check: staged file names match:"; echo "$hit"; exit 1; }
    need "$O/owens-fasta.tar.gz"; need "$O/owens_sequences.csv.gz"
    tar -tzf "$O/owens-fasta.tar.gz" | grep -i "$EXCLUDED" && { echo "exclusion check: match in owens-fasta.tar.gz"; exit 1; } || true
    n=$("$GZ" -dc "$O/owens_sequences.csv.gz" | grep -ci "$EXCLUDED" || true)
    [ "$n" = 0 ] || { echo "exclusion check: $n matching rows in owens_sequences.csv.gz"; exit 1; }
    echo "exclusion check: clean (names, FASTA tarball, sequence table)"
}

# ── Zenodo: processed data ───────────────────────────────────────────────────
Z=$STAGE/zenodo-data
for rel in \
    metadata/2024-SRARunTable-MetadataCleaning.xlsx \
    metadata/allOPIGdefinedUnpairedMetadata.csv \
    classifier/classifier_data_igblast-train-weights.csv \
    classifier/classifier_data_igblast-test-weights.csv \
    classifier/treatment_sequences_train.csv \
    classifier/treatment_sequences_test.csv \
    embedding-comparison/triplet_embedding_test_set_igblast.csv \
    foundation/diverse_foundation_data.csv \
    foundation/undiversified_foundation_data.csv \
    foundation/diverse_foundation_data_protein.csv \
    foundation/junction.csv \
    foundation/classifier_data_igh_sequence-val.csv \
    sequences/preliminary-trust4-quality-sequences.csv \
    sequences/all_high_quality_for_igblast.csv \
    sequences/all_hq_igblast_realigned.csv
do
    echo "zenodo-data: $rel"; gz_into "$Z" "$rel"
done
fresh "$Z/comparator-output.tar.gz" "$BCRM_DATA/comparator-output" || { tar "${TAR_OPTS[@]}" -C "$BCRM_DATA" -cf - comparator-output | "$GZ" > "$Z/comparator-output.tar.gz.part" && mv "$Z/comparator-output.tar.gz.part" "$Z/comparator-output.tar.gz"; }
rm -rf "$Z/owens"   # the Owens sequences are not public (packaged separately below)
cp "$REPO_ROOT/docs/release/zenodo-data-README.md" "$Z/README.md"

# ── Owens antibody sequences (Fig. 3D): not public, shared on request ───────
# A restricted-access Zenodo record or direct sharing; never part of the public records. Flat folder.
O=$STAGE/owens-on-request; mkdir -p "$O"
fresh "$O/owens_sequences.csv.gz" "$BCRM_DATA/owens/owens_sequences.csv" || {
    "$GZ" -c "$BCRM_DATA/owens/owens_sequences.csv" > "$O/owens_sequences.csv.gz.part" && mv "$O/owens_sequences.csv.gz.part" "$O/owens_sequences.csv.gz"; }
fresh "$O/owens-fasta.tar.gz" "$BCRM_DATA/owens/MS Repertoire Fasta Files" || { tar "${TAR_OPTS[@]}" -C "$BCRM_DATA/owens" -czf "$O/owens-fasta.tar.gz.part" "MS Repertoire Fasta Files" && mv "$O/owens-fasta.tar.gz.part" "$O/owens-fasta.tar.gz"; }
cp "$REPO_ROOT/docs/release/owens-on-request-README.md" "$O/README.md"
sums "$O"

# ── Zenodo: TRUST4 assemblies (SRA runs vs in-house samples) ─────────────────
T=$Z   # the TRUST4 assemblies share the data record
(cd "$BCRM_DATA/trust4" && ls TRUST4results | grep '_airr.tsv$' > "$T/all.txt")
grep -E '(SRR|ERR)[0-9]+' "$T/all.txt" | sed 's|^|TRUST4results/|' > "$T/sra-files.txt" || true
grep -vE '(SRR|ERR)[0-9]+' "$T/all.txt" | sed 's|^|TRUST4results/|' > "$T/inhouse-files.txt" || true
rm "$T/all.txt"
for part in sra inhouse; do
    echo "zenodo-data (trust4): $part ($(wc -l < "$T/$part-files.txt") files)"
    fresh "$T/trust4-$part.tar.gz" "$BCRM_DATA/trust4" || { tar "${TAR_OPTS[@]}" -C "$BCRM_DATA/trust4" -cf - -T "$T/$part-files.txt" | "$GZ" > "$T/trust4-$part.tar.gz.part" \
        && mv "$T/trust4-$part.tar.gz.part" "$T/trust4-$part.tar.gz"; }
done
cp "$REPO_ROOT/docs/release/zenodo-trust4-README.md" "$Z/README-trust4.md"
flat_ok "$Z"; sums "$Z"

# ── Hugging Face: weights, transformers config, tokenizers, trained models ───
H=$STAGE/huggingface; mkdir -p "$H/tokenizers"
# Not released: the final 500M weights (-long.pth), because the paper uses its 48 h checkpoint
# throughout, and the 104M / 304M replications (*768x11*, *1024x20*), which back no published panel
# (their training scripts and their row in foundation_models_eval_losses.csv are released instead).
unreleased() { case $(basename "$1") in *-gradClip-long.pth|*768x11*|*1024x20*) return 0 ;; esac; return 1; }
for f in "$H"/*.pth; do   # drop weights that are no longer in BCRM_MODELS, or no longer released
    [ -e "$f" ] || continue
    { [ ! -e "$BCRM_MODELS/$(basename "$f")" ] || unreleased "$f"; } && rm "$f"
done
for f in "$BCRM_MODELS"/*.pth; do
    unreleased "$f" && continue
    fresh "$H/$(basename "$f")" "$f" || cp "$f" "$H/"
done
for d in "$REPO_ROOT"/foundation-models/*tokenizer*; do cp -r "$d" "$H/tokenizers/"; done
# Trained sequence classifiers and waypoint models (reused by the seqclf/waypoints stages)
fresh "$STAGE/.trained-staged" "$BCRM_MODELS/trained" || {   # marker kept outside the upload folder
    rm -rf "$H/trained"; cp -r "$BCRM_MODELS/trained" "$H/trained"; date > "$STAGE/.trained-staged"; }
# transformers-format export = the 487M config + its 48 h weights (needs PyTorch and transformers,
# so run this on a GPU node; the script prints what to rerun if they are missing)
HF_OUT=$H/2025-HL-Foundation-1024x4096-HF; mkdir -p "$HF_OUT"
cp "$BCRM_MODELS"/2025-HL-Foundation-1024x4096-HF/*config.json "$HF_OUT/"
export_cmd=(python "$REPO_ROOT/foundation-models/export_hf.py" --config "$BCRM_MODELS/2025-HL-Foundation-1024x4096-HF"
            --weights "$BCRM_MODELS/2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth" --out "$HF_OUT")
if [ -s "$HF_OUT/model.safetensors" ]; then :
elif python -c "import torch, transformers" 2>/dev/null; then "${export_cmd[@]}"
else echo "NOTE: run this where PyTorch is available, then rerun this script:"; echo "  ${export_cmd[*]}"; fi
cp "$REPO_ROOT/docs/release/huggingface-README.md" "$H/README.md"   # model card
cp "$REPO_ROOT/LICENSE" "$H/LICENSE"   # the weights are released under the repository's terms
sums "$H"

no_excluded

du -sh "$Z" "$H"
echo "Staged in $STAGE (not uploaded)."
