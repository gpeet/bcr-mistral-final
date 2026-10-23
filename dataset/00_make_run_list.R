# Build the TRUST4 run list (03_trust4_sra.sh input) from the curated SRA run table.
#
# Usage:
#   Rscript dataset/00_make_run_list.R <output_csv>
#
# Writes a CSV with columns run,layout,mode: one row per SRA run that has an accession and a
# library layout (PAIRED/SINGLE). Reads metadata/2024-SRARunTable-MetadataCleaning.xlsx
# under BCRM_DATA. mode is standard unless dataset/trust4_run_modes.csv says otherwise:
#   qc  - runs that timed out in the standard pass; rerun after fastp QC with --repseq
#   10x - large 5' single-cell runs (PRJNA549712) rerun with barcode/UMI-aware assembly

source(here::here("R", "paths.R"))
source(here::here("R", "metadata.R"))
suppressPackageStartupMessages(library(tidyverse))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript dataset/00_make_run_list.R <output_csv>")

runs <- read_metadata() %>%
  filter(!is.na(Run), LibraryLayout %in% c("PAIRED", "SINGLE")) %>%
  distinct(run = Run, layout = LibraryLayout) %>%
  left_join(read_csv(here::here("dataset", "trust4_run_modes.csv"), show_col_types = FALSE), by = "run") %>%
  mutate(mode = replace_na(mode, "standard"))

dir.create(dirname(args[1]), recursive = TRUE, showWarnings = FALSE)
write_csv(runs, args[1])
cat(nrow(runs), "runs written to", args[1], "\n")
