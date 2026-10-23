#!/bin/bash
set -euo pipefail

# Submit pipeline stages to Slurm on CU Boulder Alpine as a dependency chain.
#
# Usage (from the repo root, on a login node):
#   workflow/submit_alpine.sh env           # build the Python (GPU) and R environments
#   workflow/submit_alpine.sh downstream    # embeddings -> classifiers -> waypoints -> figures
#   workflow/submit_alpine.sh gpu           # the GPU part of downstream only (no R stages)
#   workflow/submit_alpine.sh figures       # comparators -> patient classifier -> SHAP -> figures
#   workflow/submit_alpine.sh fig3b-vj      # the V-J clipped variant of Fig. 3B, end to end
#   workflow/submit_alpine.sh <stage> [args]  # any single stage from workflow/run.sh list
#
# Needs workflow/config.sh with BCRM_SITE=alpine. Logs go to $BCRM_OUT/logs.
# Other Slurm clusters: copy this file and change the partition/QoS/GRES lines below.
# Each job is one process with --cpus-per-task cores (what R's parallelly and the scripts read).

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO_ROOT/workflow/config.sh"
mkdir -p "$BCRM_OUT/logs"

GPU=(--partition=ami100 --qos=gpu-normal --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:mi100:1 --time=24:00:00)
GPU_LONG=(--partition=ami100 --qos=gpu-long --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:mi100:1 --time=7-00:00:00)
GPU_TEST=(--partition=ami100 --qos=gpu-testing --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:mi100:1 --time=01:00:00)
CPU=(--partition=acpu --qos=cpu-normal --nodes=1 --ntasks=1 --cpus-per-task=16 --time=12:00:00)
COMPILE=(--partition=acompile --qos=compile --nodes=1 --ntasks=1 --cpus-per-task=4 --time=12:00:00)

# submit <after-job-ids|-> <stage> <sbatch resource array name> [stage args]
submit() {
    local after=$1 stage=$2; local -n res=$3; shift 3
    local dep=()   # <after>: job ids (afterok, or DEP_TYPE), or a full Slurm spec starting with "after"
    if [[ $after == after* ]]; then dep=(--dependency="$after")
    elif [ "$after" != "-" ]; then dep=(--dependency="${DEP_TYPE:-afterok}:$after"); fi
    sbatch --parsable --export=NONE "${res[@]}" "${dep[@]}" \
        --job-name="bcrm-$stage" --output="$BCRM_OUT/logs/%x-%j.log" \
        --wrap="bash -l '$REPO_ROOT/workflow/run.sh' $stage $*"
}

target=${1:?see usage in $0}; shift || true
case "$target" in
env)
    py=$(submit - env-python GPU_TEST)
    r=$(submit - env-r COMPILE)
    t=$(DEP_TYPE=afterany submit "$r" env-tools COMPILE)   # after env-r: concurrent conda creates race on the package cache
    echo "env-python $py  env-r $r  env-tools $t" ;;
figures)
    c=$(submit - comparators CPU)
    p=$(submit "$c" patient CPU)
    s=$(submit "$p" shap GPU)
    f=$(submit "$s" figures CPU)
    echo "comparators $c  patient $p  shap $s  figures $f" ;;
fig3b-vj)
    # The V-J clipped variant of Fig. 3B, end to end (new files only; the contig results are untouched)
    d=$(submit - treatment-vj-dataset CPU)
    e=$(submit "$d" embed GPU treatment-vj)
    t=$(submit "$e" treatment-vj GPU)
    f=$(submit "$t" treatment-vj-figures CPU)
    echo "treatment-vj-dataset $d  embed $e  treatment-vj $t  figures $f" ;;
gpu|downstream)
    # One job per heavy script so each fits in a 24 h GPU job
    e_tr=$(submit - embed GPU main-train); e_te=$(submit - embed GPU main-test); e_main="$e_tr:$e_te"
    e_junc=$(submit - embed GPU junction)
    e_trt=$(submit - embed GPU treatment); e_ow=$(submit - embed GPU owens)
    ec=$(submit - embed-compare GPU)
    # The two classifiers on the same embeddings share a cache in $BCRM_OUT/preprocessed: run them in turn
    k1=$(submit "$e_main" seqclf GPU exclusion); k2=$(submit "afterok:$e_main,afterany:$k1" seqclf GPU baseline)
    k3=$(submit "$e_junc" seqclf GPU junction-exclusion); k4=$(submit "afterok:$e_junc,afterany:$k3" seqclf GPU junction-baseline)
    w=$(submit "$k1" waypoints GPU)
    t=$(submit "$k1:$k2:$e_trt" treatment GPU)
    o=$(submit "$k1:$k2:$k3:$k4:$e_ow" owens GPU)
    echo "embed $e_tr $e_te $e_junc $e_trt $e_ow  embed-compare $ec  seqclf $k1 $k2 $k3 $k4  waypoints $w  treatment $t  owens $o"
    if [ "$target" = downstream ]; then
        c=$(submit - comparators CPU)
        p=$(submit "$w:$c" patient CPU)
        s=$(submit "$p" shap GPU)
        f=$(submit "$s:$t:$o:$ec" figures CPU)
        echo "comparators $c  patient $p  shap $s  figures $f"
    fi ;;
trust4)
    # trust4 [standard|qc|10x]: one array job per mode, with the resources each mode needed
    mode=${1:-standard}
    case "$mode" in
        standard) res=(--partition=acpu --qos=cpu-normal --ntasks=1 --cpus-per-task=8 --mem=32G --time=22:00:00) ;;
        qc)       res=(--partition=acpu --qos=cpu-long --ntasks=1 --cpus-per-task=8 --mem=32G --time=2-00:00:00) ;;
        10x)      res=(--partition=amem --qos=mem-long --ntasks=1 --cpus-per-task=8 --mem=512G --time=2-00:00:00) ;;
        *) echo "mode must be standard, qc or 10x"; exit 1 ;;
    esac
    idx=$(awk -F, -v m="$mode" 'NR > 1 && $3 == m { printf "%s%d", sep, NR - 1; sep = "," }' "$BCRM_DATA/trust4/sra_runs.csv")
    [ -n "$idx" ] || { echo "no $mode runs in sra_runs.csv"; exit 1; }
    sbatch --parsable --export=NONE --nodes=1 "${res[@]}" --array="$idx%100" --job-name="bcrm-trust4-$mode" \
        --output="$BCRM_OUT/logs/%x-%A_%a.log" \
        --wrap="bash -l '$REPO_ROOT/workflow/run.sh' trust4" ;;
igblast)
    n=$(ls "$BCRM_DATA"/trust4/TRUST4results/*_airr.tsv | wc -l)
    sbatch --parsable --export=NONE "${CPU[@]/--cpus-per-task=16/--cpus-per-task=4}" --array="1-$(( (n + 49) / 50 ))" \
        --job-name=bcrm-igblast --output="$BCRM_OUT/logs/%x-%A_%a.log" \
        --wrap="bash -l '$REPO_ROOT/workflow/run.sh' igblast" ;;
env-python)
    submit - env-python GPU_TEST ;;     # CURC's PyTorch module only loads on GPU nodes
env-r|env-tools)
    submit - "$target" COMPILE ;;
pretrain)
    # The large models train for days (the paper uses the 48 h checkpoint of the 500M model)
    case "${1:-}" in 500M|500M_junction) resources=GPU_LONG ;; *) resources=GPU ;; esac
    submit - pretrain "$resources" "$@" ;;
embed|embed-compare|seqclf|waypoints|treatment|treatment-vj|owens|shap|tune|eval-foundation|diversity)
    submit - "$target" GPU "$@" ;;
*)
    submit - "$target" CPU "$@" ;;
esac
