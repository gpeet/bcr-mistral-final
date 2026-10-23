#!/bin/bash
set -euo pipefail

# Build the master OPIG table: per file, productive complete-VDJ IGH sequences, the
# 25,000 with the highest Redundancy, tagged with the source file name.
#
# Usage:
#   ./06_master_opig.sh <opig_dir> <output_csv> [threads]
#
# Arguments:
#   opig_dir    - directory of OPIG .csv.gz files (01_download_opig.sh output)
#   output_csv  - e.g. $BCRM_DATA/sequences/master-opig-dataset-sequence.csv
#   threads     - parallel jobs (default: 16)
#
# Requires csvtk and GNU parallel (envs/environment-tools.yml).

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <opig_dir> <output_csv> [threads]"
    exit 1
fi

INDIR=$1
OUT=$2
THREADS=${3:-16}
mkdir -p "$(dirname "$OUT")"

echo "sequence,productive,complete_vdj,junction,junction_aa,consensus_count,v_call,d_call,j_call,v_identity,file_name" > "$OUT"

process_file() {
  local file="$1"
  local base
  base=$(basename "$file")

  zcat "$file" \
    | tail -n +2 \
    | csvtk --num-cpus 1 cut -f sequence,productive,complete_vdj,junction,junction_aa,Redundancy,v_call,d_call,j_call,v_identity \
    | csvtk --num-cpus 1 grep -f complete_vdj -p "T" \
    | csvtk --num-cpus 1 grep -f productive -p "T" \
    | csvtk --num-cpus 1 grep -f v_call -r -p "^IGH" \
    | csvtk --num-cpus 1 sort -k Redundancy:nr \
    | sed 1d \
    | head -n 25000 \
    | awk -v fname="$base" -F',' 'BEGIN{OFS=","} { print $0, fname }'
}

export -f process_file

find "$INDIR" -name '*.csv.gz' \
  | parallel --bar -j "$THREADS" process_file {} >> "$OUT"
