# Maintainer tool: write renv.lock from the R library that produced the published results,
# recording the version of every package the repository's code uses.
#
# Usage (from the repository root, with renv installed):
#   Rscript envs/make_renv_lock.R <library_path>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript envs/make_renv_lock.R <library_path>")

.libPaths(c(args[1], .libPaths()))  # also used to parse the notebooks' dependencies

# force: igblastr was installed from GitHub (bioc/igblastr), so renv sees it as coming from
# a different Bioconductor release than the rest; its remote SHA is still recorded.
renv::snapshot(
  project  = getwd(),
  library  = .libPaths(),
  lockfile = "renv.lock",
  type     = "implicit",
  prompt   = FALSE,
  force    = TRUE
)
