#!/bin/bash
set -euo pipefail

# Install the R packages pinned in renv.lock into the project library.
# Run inside the bcrm-r conda environment (envs/environment-r.yml).
#
# Environment:
#   RENV_PATHS_ROOT  where renv keeps its package cache (default: renv's own default)

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
Rscript -e 'renv::restore(prompt = FALSE)'
