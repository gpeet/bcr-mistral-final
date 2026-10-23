# Copy to workflow/config.sh (gitignored) and edit. Sourced by workflow/run.sh.

# Where the inputs, model weights and outputs live (see docs/DATA_LAYOUT.md)
export BCRM_DATA=/path/to/bcrm/data
export BCRM_MODELS=/path/to/bcrm/models
export BCRM_OUT=/path/to/bcrm/results

# Site file in workflow/site/ that knows how to activate the environments
export BCRM_SITE=generic        # or: alpine

# pip --target directory for the GPU Python packages (envs/setup_python_env.sh)
export BCRM_PYLIBS=/path/to/bcrm-pylibs

# conda environments for the R stages and the dataset tools (envs/*.yml)
export BCRM_R_ENV=bcrm-r
export BCRM_TOOLS_ENV=bcrm-tools

# Only needed to rebuild the dataset
# export TRUST4_HOME=/path/to/TRUST4
# export IGBLASTR_CACHE=/path/to/cache/R/igblastr   # must end in R/igblastr
