#!/bin/bash
set -euo pipefail

# Download the released data and weights into BCRM_DATA / BCRM_MODELS, checking SHA-256 sums.
#
# Usage:
#   scripts/fetch_data.sh [data] [trust4] [models] [owens]     (default: data models)
#
#   data    processed data from Zenodo (~5 GB unpacked): every figure except 1E, Table S1, 1I and 3D
#   trust4  TRUST4 assemblies, in the same Zenodo record (~47 GB unpacked): Fig. 1I; with
#           `run.sh master-trust4`, Fig. 1E and Table S1; and rebuilding the dataset
#   models  BCR-Mistral weights and trained classifiers from the Hugging Face Hub (~10 GB)
#   owens   Owens et al. antibody sequences (Fig. 3D): not public, shared on request; unpacks the
#           folder given in OWENS_BUNDLE
#
# Needs curl and sha256sum; `models` needs Python with huggingface_hub (pip install huggingface_hub).

# The processed data and TRUST4 assemblies share one Zenodo record (CC BY 4.0). The model weights are
# on the Hugging Face Hub instead, under the repository's own terms; see LICENSE.
ZENODO_DATA_RECORD=${ZENODO_DATA_RECORD:-23605219}   # v1.0.1; 10.5281/zenodo.20432912 is all versions
HF_REPO=${HF_REPO:-grpeet/bcr-mistral}           # access is gated: accept the terms on the model page first

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BCRM_DATA=${BCRM_DATA:-$REPO_ROOT/data}
BCRM_MODELS=${BCRM_MODELS:-$REPO_ROOT/models}

zenodo_fetch() {  # zenodo_fetch <record> <download dir>: every file listed in the record's SHA256SUMS
    local record=${1:?Zenodo record id not set; see the top of $0} dir=$2
    local url="https://zenodo.org/records/$record/files"
    mkdir -p "$dir"
    curl -fsSL "$url/SHA256SUMS?download=1" -o "$dir/SHA256SUMS"
    while read -r _ f; do
        f=${f#./}
        [ -s "$dir/$f" ] && continue
        mkdir -p "$dir/$(dirname "$f")"
        echo "download $f"
        curl -fSL "$url/$(basename "$f")?download=1" -o "$dir/$f"
    done < "$dir/SHA256SUMS"
    (cd "$dir" && sha256sum -c --quiet SHA256SUMS)
}

parts=("$@"); [ ${#parts[@]} -gt 0 ] || parts=(data models)
for part in "${parts[@]}"; do
    case "$part" in
    data)
        dl=$BCRM_DATA/.download/data
        zenodo_fetch "$ZENODO_DATA_RECORD" "$dl"
        (cd "$dl" && find . -name '*.gz' ! -name '*.tar.gz') | while read -r f; do
            out="$BCRM_DATA/${f%.gz}"; mkdir -p "$(dirname "$out")"
            [ "$out" -nt "$dl/$f" ] || { gunzip -c "$dl/$f" > "$out.part" && mv "$out.part" "$out"; }   # new or updated
        done
        tar -C "$BCRM_DATA" -xzf "$dl/comparator-output.tar.gz" ;;
    trust4)   # the same record as `data`; only the TRUST4 tarballs are unpacked here
        dl=$BCRM_DATA/.download/data
        zenodo_fetch "$ZENODO_DATA_RECORD" "$dl"
        mkdir -p "$BCRM_DATA/trust4"
        for t in "$dl"/trust4-*.tar.gz; do tar -C "$BCRM_DATA/trust4" -xzf "$t"; done ;;
    models)
        : "${HF_REPO:?Hugging Face repo id not set; see the top of $0}"
        python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' \
            "$HF_REPO" "$BCRM_MODELS"
        (cd "$BCRM_MODELS" && sha256sum -c --quiet SHA256SUMS) ;;
    owens)  # the Owens antibody sequences (Fig. 3D) are shared on request: OWENS_BUNDLE = folder you received
        src=${OWENS_BUNDLE:?set OWENS_BUNDLE to the folder holding the Owens files (shared on request; see README)}
        (cd "$src" && sha256sum -c --quiet SHA256SUMS)
        mkdir -p "$BCRM_DATA/owens"
        gunzip -c "$src/owens_sequences.csv.gz" > "$BCRM_DATA/owens/owens_sequences.csv"
        tar -C "$BCRM_DATA/owens" -xzf "$src/owens-fasta.tar.gz" ;;
    *) echo "unknown part: $part (data, trust4, models, owens)"; exit 1 ;;
    esac
done
if [ -d "$BCRM_DATA/.download" ]; then
    echo "Compressed downloads are kept in $BCRM_DATA/.download (delete it to save space; only needed to re-check or re-unpack)."
fi
