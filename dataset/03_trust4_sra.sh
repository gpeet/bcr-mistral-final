#!/bin/bash
set -euo pipefail

# Download one SRA run and assemble BCR/TCR sequences with TRUST4.
#
# Usage:
#   ./03_trust4_sra.sh <run_list> [index]
#
# Arguments:
#   run_list - CSV with header "run,layout,mode" (see 00_make_run_list.R)
#   index    - 1-based row of run_list to process; defaults to $SLURM_ARRAY_TASK_ID
#              so the script can run directly as a Slurm array task
#
# Modes (dataset/trust4_run_modes.csv):
#   standard - TRUST4 on the raw reads
#   qc       - fastp quality trimming + deduplication first, TRUST4 with --repseq
#              (the runs that did not finish in the standard pass; allow ~2 days)
#   10x      - fastp, then barcode/UMI-aware TRUST4 (read 1 = 16 nt barcode + UMI;
#              needs a high-memory node, ~512 GB)
#
# Environment:
#   TRUST4_HOME  directory containing run-trust4, hg38_bcrtcr.fa, human_IMGT+C.fa (required)
#   BCRM_DATA    output root; results go to $BCRM_DATA/trust4/TRUST4results (default: <repo>/data)
#   WORK_DIR     scratch space for .sra/.fastq files (default: $BCRM_DATA/trust4/work)
#   THREADS      TRUST4/fastp/fasterq-dump threads (default: $SLURM_CPUS_PER_TASK, $SLURM_NTASKS or 4)
#   MAX_SPOTS    smoke test: stream only the first N spots (fastq-dump -X) instead of the whole run
#
# Requires sra-tools and fastp on PATH (envs/environment-tools.yml).

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <run_list> [index]"
    exit 1
fi

run_list=$1
index=${2:-${SLURM_ARRAY_TASK_ID:?give an index or run as a Slurm array task}}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BCRM_DATA=${BCRM_DATA:-$REPO_ROOT/data}
WORK_DIR=${WORK_DIR:-$BCRM_DATA/trust4/work}
OUT_DIR="$BCRM_DATA/trust4/TRUST4results"
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-4}}}
: "${TRUST4_HOME:?set TRUST4_HOME to your TRUST4 install}"

# Row index+1 skips the header line
IFS=, read -r sraName sraLayout sraMode < <(sed -n "$((index + 1))p" "$run_list" | tr -d '\r')
if [ -z "${sraName:-}" ]; then
    echo "No run at index $index in $run_list"
    exit 1
fi
sraMode=${sraMode:-standard}
echo "Run $index: $sraName ($sraLayout, $sraMode)"

FQ="$WORK_DIR/fastqs"
mkdir -p "$WORK_DIR/sra" "$FQ" "$OUT_DIR"
cd "$WORK_DIR"   # fastp writes fastp.json/fastp.html here

# ── Download ────────────────────────────────────────────────────────────────
if [ -n "${MAX_SPOTS:-}" ]; then
    fastq-dump --split-3 --skip-technical -X "$MAX_SPOTS" -O "$FQ" "$sraName"
else
    prefetch --max-size 500G -O "$WORK_DIR/sra" "$sraName"
    sraFile=$(find "$WORK_DIR/sra" -name "${sraName}.sra" | head -1)

    fasterq-dump \
        --split-3 \
        --skip-technical \
        --mem 32G \
        --threads "$THREADS" \
        -O "$FQ" \
        "$sraFile"

    rm -f "$sraFile"
fi

if [[ "$sraLayout" == "PAIRED" ]]; then
    r1="$FQ/${sraName}_1.fastq"; r2="$FQ/${sraName}_2.fastq"
else
    r1="$FQ/${sraName}.fastq"; r2=""
fi

# ── Read QC (qc and 10x modes) ───────────────────────────────────────────────
extra=()
case "$sraMode" in
standard) ;;
qc)
    if [ -n "$r2" ]; then
        fastp -i "$r1" -I "$r2" -o "$FQ/fastp_${sraName}_1.fq" -O "$FQ/fastp_${sraName}_2.fq" \
            --dedup --thread "$THREADS" --qualified_quality_phred 30 --cut_front \
            --cut_mean_quality 30 --unqualified_percent_limit 10 --n_base_limit 0
        r1="$FQ/fastp_${sraName}_1.fq"; r2="$FQ/fastp_${sraName}_2.fq"
    else
        fastp -i "$r1" -o "$FQ/fastp_${sraName}.fq" \
            --dedup --thread "$THREADS" --qualified_quality_phred 30 --cut_front --cut_tail \
            --cut_mean_quality 30 --unqualified_percent_limit 10 --n_base_limit 0
        r1="$FQ/fastp_${sraName}.fq"
    fi
    extra=(--repseq) ;;
10x)
    fastp -i "$r1" -I "$r2" -o "$FQ/fastp_${sraName}_1.fq" -O "$FQ/fastp_${sraName}_2.fq" \
        --thread "$THREADS" --dedup --qualified_quality_phred 30 \
        --unqualified_percent_limit 5 --n_base_limit 0
    r1="$FQ/fastp_${sraName}_1.fq"; r2="$FQ/fastp_${sraName}_2.fq"
    extra=(--barcode "$r1" --UMI "$r1" --readFormat bc:0:15,um:16:26) ;;
*)
    echo "Unknown mode: $sraMode"; exit 1 ;;
esac

# ── TRUST4 ──────────────────────────────────────────────────────────────────
if [ -n "$r2" ]; then
    reads=(-1 "$r1" -2 "$r2")
else
    reads=(-u "$r1")
fi

"$TRUST4_HOME/run-trust4" \
    "${reads[@]}" \
    -f "$TRUST4_HOME/hg38_bcrtcr.fa" \
    --ref "$TRUST4_HOME/human_IMGT+C.fa" \
    --od "$OUT_DIR" \
    -t "$THREADS" \
    ${extra[@]+"${extra[@]}"} \
    --clean 2

# ── Clean up fastqs and TRUST4's candidate-read files ────────────────────────
rm -f "$FQ/${sraName}"*.fastq "$FQ/fastp_${sraName}"*.fq "$OUT_DIR"/TRUST_*"${sraName}"*_toassemble*
