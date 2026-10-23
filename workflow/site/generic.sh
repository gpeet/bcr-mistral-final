# Generic site: assumes python (with torch 1.13.0 importable) and Rscript are on PATH.
# Copy this file to workflow/site/<name>.sh and set BCRM_SITE=<name> to adapt it.

bcrm_env_gpu_base() {
    export PYTHONNOUSERSITE=1
}

bcrm_env_gpu() {
    bcrm_env_gpu_base
    export PYTHONPATH="$REPO_ROOT${BCRM_PYLIBS:+:$BCRM_PYLIBS}:${PYTHONPATH:-}"
}

bcrm_env_r() { :; }

bcrm_env_tools() { :; }   # needs the tools in envs/environment-tools.yml on PATH

bcrm_conda() { :; }       # conda must already be on PATH
