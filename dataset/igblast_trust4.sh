#!/bin/bash
set -euo pipefail

# Re-annotate TRUST4 AIRR files with the IgBLAST CLI.
# Each call processes one batch of BATCH_SIZE AIRR files (the last batch gets the remainder).
#
# Uses igblastn 1.22.0 with:
#   Germline: IMGT release 202531-1  (Homo sapiens IGH+IGK+IGL)
#   C region: IMGT release 202412    (human IGH+IGK+IGL)
# These are the databases installed by the igblastr R package (dataset/build_dataset.Rmd).
#
# Usage:
#   ./igblast_trust4.sh [batch]      batch is 1-based; defaults to $SLURM_ARRAY_TASK_ID
#
# Environment:
#   BCRM_DATA     input  $BCRM_DATA/trust4/TRUST4results/*_airr.tsv
#                 output $BCRM_DATA/trust4/igblast_results/<accession>_igblast.tsv  (default: <repo>/data)
#   IGBLASTR_CACHE igblastr cache dir (default: ${XDG_CACHE_HOME:-$HOME/.cache}/R/igblastr)
#   THREADS       igblastn threads (default: $SLURM_CPUS_PER_TASK, $SLURM_NTASKS or 4)
#   BATCH_SIZE    AIRR files per batch (default: 50)

# --- Paths -------------------------------------------------------------------
TASK=${1:-${SLURM_ARRAY_TASK_ID:?give a batch number or run as a Slurm array task}}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BCRM_DATA=${BCRM_DATA:-$REPO_ROOT/data}
TRUST4_DIR="$BCRM_DATA/trust4/TRUST4results"
OUT_DIR="$BCRM_DATA/trust4/igblast_results"
WORK_DIR="$BCRM_DATA/trust4/igblast_tmp/task_${TASK}"

IGBLASTR_CACHE=${IGBLASTR_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/R/igblastr}
IGBLAST_ROOT="$IGBLASTR_CACHE/igblast_roots/1.22.0"
IGBLASTN="$IGBLAST_ROOT/bin/igblastn"
GERMLINE_DB="$IGBLASTR_CACHE/germline_dbs/IMGT-202531-1.Homo_sapiens.IGH+IGK+IGL"
C_REGION_DB="$IGBLASTR_CACHE/c_region_dbs/_IMGT.human.IGH+IGK+IGL.202412/C"

IMGT_GERMLINE_RELEASE="202531-1"
IMGT_C_REGION_RELEASE="202412"

BATCH_SIZE=${BATCH_SIZE:-50}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-4}}}

export IGDATA="$IGBLAST_ROOT"

mkdir -p "$OUT_DIR" "$WORK_DIR"

# --- Build file list for this task -------------------------------------------
mapfile -t ALL_FILES < <(ls "$TRUST4_DIR"/*_airr.tsv 2>/dev/null | sort)
N_FILES=${#ALL_FILES[@]}

START=$(( (TASK - 1) * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))
if [ $END -gt $N_FILES ]; then END=$N_FILES; fi
if [ $START -ge $N_FILES ]; then
    echo "Batch $TASK: no files to process (start=$START >= total=$N_FILES)"
    exit 0
fi

TASK_FILES=("${ALL_FILES[@]:$START:$((END - START))}")
echo "Batch $TASK: processing files $((START+1))-$END of $N_FILES (${#TASK_FILES[@]} files)"

# --- Process each file -------------------------------------------------------
for AIRR_FILE in "${TASK_FILES[@]}"; do
    BASENAME=$(basename "$AIRR_FILE" _airr.tsv)
    ACCESSION=$(echo "$BASENAME" | grep -oP '(SRR|ERR)[0-9]+' || echo "$BASENAME")
    OUTPUT_FILE="$OUT_DIR/${ACCESSION}_igblast.tsv"

    # Skip if already done
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  $ACCESSION: already exists, skipping"
        continue
    fi

    # Count sequences (subtract header)
    N_SEQ=$(( $(wc -l < "$AIRR_FILE") - 1 ))
    if [ "$N_SEQ" -le 0 ]; then
        echo "  $ACCESSION: empty, skipping"
        continue
    fi

    # Extract sequences to FASTA
    FASTA_FILE="$WORK_DIR/${ACCESSION}.fasta"
    awk -F'\t' 'NR > 1 && $2 != "" {
        print ">" $1
        print $2
    }' "$AIRR_FILE" > "$FASTA_FILE"

    # Run igblastn
    "$IGBLASTN" \
        -germline_db_V "$GERMLINE_DB/V" \
        -germline_db_D "$GERMLINE_DB/D" \
        -germline_db_J "$GERMLINE_DB/J" \
        -c_region_db "$C_REGION_DB" \
        -auxiliary_data "$IGBLAST_ROOT/optional_file/human_gl.aux" \
        -organism human \
        -ig_seqtype Ig \
        -domain_system imgt \
        -outfmt 19 \
        -num_alignments_V 1 \
        -num_alignments_D 1 \
        -num_alignments_J 1 \
        -num_alignments_C 1 \
        -num_threads "$THREADS" \
        -query "$FASTA_FILE" \
        -out "$WORK_DIR/${ACCESSION}_raw.tsv"

    # Add provenance columns: sra_accession, imgt_germline_release, imgt_c_region_release
    awk -v acc="$ACCESSION" \
        -v grel="$IMGT_GERMLINE_RELEASE" \
        -v crel="$IMGT_C_REGION_RELEASE" \
        'BEGIN {FS=OFS="\t"}
         NR==1 {print $0, "sra_accession", "imgt_germline_release", "imgt_c_region_release"; next}
         {print $0, acc, grel, crel}' \
        "$WORK_DIR/${ACCESSION}_raw.tsv" > "$OUTPUT_FILE"

    echo "  $ACCESSION: $N_SEQ sequences annotated"

    # Clean up temp files
    rm -f "$FASTA_FILE" "$WORK_DIR/${ACCESSION}_raw.tsv"
done

# Clean up work directory
rm -rf "$WORK_DIR"
echo "Batch $TASK: done"
