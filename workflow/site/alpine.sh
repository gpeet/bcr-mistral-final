# CU Boulder Alpine: GPU stages run on AMD MI100 nodes (ami100) with CURC's ROCm build of
# PyTorch 1.13.0; R stages run in the bcrm-r conda environment (envs/environment-r.yml).
# Needs BCRM_PYLIBS and BCRM_R_ENV from workflow/config.sh.

bcrm_env_gpu_base() {
    module purge
    module load rocm/5.2.3 pytorch/1.13.0
    export PYTHONNOUSERSITE=1          # keep ~/.local packages out
    export HF_HOME=${HF_HOME:-$BCRM_OUT/.hf_cache}
}

bcrm_env_gpu() {
    bcrm_env_gpu_base
    export PYTHONPATH="$REPO_ROOT${BCRM_PYLIBS:+:$BCRM_PYLIBS}${PYTHONPATH:+:$PYTHONPATH}"
}

bcrm_env_r() {
    module purge
    module load miniforge
    conda activate "$BCRM_R_ENV"
    # CURC's miniforge module points XDG_CACHE_HOME at the pip cache, which would hide the igblastr
    # germline databases from R; keep R's package caches next to IGBLASTR_CACHE (<dir>/R/igblastr)
    if [ -n "${IGBLASTR_CACHE:-}" ]; then
        [[ $IGBLASTR_CACHE == */R/igblastr ]] || { echo "IGBLASTR_CACHE must end in /R/igblastr (igblastr's cache layout)" >&2; return 1; }
        export R_USER_CACHE_DIR=${IGBLASTR_CACHE%/R/igblastr}
    fi
}

bcrm_env_tools() {             # dataset assembly: SRA download, TRUST4, OPIG, master tables
    module purge
    module load miniforge
    conda activate "$BCRM_TOOLS_ENV"
}

bcrm_conda() {                 # make conda available (for creating environments)
    module purge
    module load miniforge
}
