# Locate bcr-mistral data, model weights and outputs (mirrors bcrm/paths.py).
# Override with BCRM_DATA, BCRM_MODELS, BCRM_OUT; defaults are data/, models/,
# results/ inside the repository. See docs/DATA_LAYOUT.md.

.bcrm_root <- function(var, default) {
  path.expand(Sys.getenv(var, unset = here::here(default)))
}

repo_path  <- function(rel = "") here::here(rel)
data_path  <- function(rel = "") file.path(.bcrm_root("BCRM_DATA", "data"), rel)
model_path <- function(rel = "") file.path(.bcrm_root("BCRM_MODELS", "models"), rel)
out_path   <- function(rel = "") {
  p <- file.path(.bcrm_root("BCRM_OUT", "results"), rel)
  # `rel` may name several outputs at once, and dir.create() takes one path at a time
  for (d in unique(dirname(p))) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  p
}
