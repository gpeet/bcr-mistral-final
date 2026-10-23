#!/bin/bash
set -euo pipefail

# Build the master TRUST4 table from per-run AIRR files: selected columns, tagged with
# the source file name. No productivity/completeness filtering (that happens downstream).
#
# Usage:
#   ./07_master_trust4.sh <trust4_dir> <output_tsv> [--chains IGH|all] [--seq-col COL]
#                         [--max-per-file N] [threads]
#
# Arguments:
#   trust4_dir  - directory of *airr.tsv files (03_trust4_sra.sh output)
#   output_tsv  - e.g. $BCRM_DATA/sequences/master-trust4-dataset-sequence-unfiltered.csv
#                 (tab-separated despite the .csv name, as in the original pipeline)
#   --chains    - IGH keeps heavy chains only; all keeps every BCR/TCR chain (default: all)
#   --seq-col   - AIRR column written as the first column (default: sequence, the full contig)
#   --max-per-file - keep only the N rows with the highest consensus_count per file (default: all)
#   threads     - parallel jobs (default: 16)
#
# The default output is the table the Fig. 1 notebooks read. The earlier (June 2025) table
# behind the Fig. 1E counts in the text was built with --seq-col sequence_alignment
# --max-per-file 10000; rebuilding it this way reproduces every row above the per-file
# cutoff, but rows tied at the cutoff count can differ (their order was not recorded).
#
# Requires csvtk and GNU parallel (envs/environment-tools.yml).

if [ $# -lt 2 ]; then
    echo "Usage: $0 <trust4_dir> <output_tsv> [--chains IGH|all] [threads]"
    exit 1
fi

INDIR=$1
OUT=$2
shift 2
CHAINS=all
SEQ_COL=sequence
MAX_PER_FILE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --chains)       CHAINS=${2:?--chains needs IGH or all}; shift 2 ;;
        --seq-col)      SEQ_COL=${2:?--seq-col needs a column name}; shift 2 ;;
        --max-per-file) MAX_PER_FILE=${2:?--max-per-file needs a number}; shift 2 ;;
        *)              break ;;
    esac
done
THREADS=${1:-16}
mkdir -p "$(dirname "$OUT")"

printf '%s\tproductive\tcomplete_vdj\tjunction\tjunction_aa\tconsensus_count\tv_call\td_call\tj_call\tc_call\tv_identity\tfile_name\n' "$SEQ_COL" > "$OUT"

process_file() {
  local file="$1"
  local base
  base=$(basename "$file")

  cat "$file" \
    | csvtk --num-cpus 1 -t cut -f "$SEQ_COL",productive,complete_vdj,junction,junction_aa,consensus_count,v_call,d_call,j_call,c_call,v_identity \
    | if [ "$CHAINS" = IGH ]; then csvtk --num-cpus 1 -t grep -f v_call -r -p "^IGH"; else cat; fi \
    | if [ "$MAX_PER_FILE" -gt 0 ]; then csvtk --num-cpus 1 -t sort -k consensus_count:nr; else cat; fi \
    | sed 1d \
    | if [ "$MAX_PER_FILE" -gt 0 ]; then head -n "$MAX_PER_FILE"; else cat; fi \
    | awk -v fname="$base" -F ',' 'BEGIN{OFS="\t"} { print $0, fname }'
}

export -f process_file
export CHAINS SEQ_COL MAX_PER_FILE

find "$INDIR" -name '*airr.tsv' \
  | parallel --bar -j "$THREADS" process_file {} >> "$OUT"
