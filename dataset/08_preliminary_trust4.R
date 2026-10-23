# Build the quality-filtered TRUST4 IGH table behind Fig. 1F-H
# (sequences/preliminary-trust4-quality-sequences.csv).
#
# SRA runs: IGH rows with consensus_count > 2 from each TRUST4 AIRR file (one per run), joined to the run
# metadata. In-house (Bennett) samples: the same filter on the master TRUST4 table, which also
# holds the non-SRA files. Rows without metadata are kept; the notebooks drop them.
#
# Usage:
#   Rscript dataset/08_preliminary_trust4.R <TRUST4results dir> <master-trust4-dataset-sequence-unfiltered.csv> \
#       <2024-SRARunTable-MetadataCleaning.xlsx> <output_csv>

suppressPackageStartupMessages({library(tidyverse); library(vroom)})
source(here::here("R", "trust4_files.R"))
source(here::here("R", "metadata.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: Rscript dataset/08_preliminary_trust4.R <trust4_dir> <master_unfiltered> <metadata_xlsx> <output_csv>")
trust4_dir <- args[1]; master_unfiltered <- args[2]; metadata_xlsx <- args[3]; out_csv <- args[4]

COLS <- c("BioProject", "Disease", "Disease Class", "DonorID", "Age", "Sex", "Tissue", "v_call", "junction_aa", "consensus_count")

metadata <- read_metadata(metadata_xlsx) %>%
  select(Run, BioProject, Tissue, Age, Sex, DonorID, Disease, `Disease Class`, Filename)

sra <- list.files(trust4_dir, full.names = TRUE, pattern = "airr.tsv") %>%
  keep_one_file_per_run() %>%   # one file per SRA run (R/trust4_files.R)
  map(vroom, id = "SRA", col_select = c("v_call", "junction_aa", "consensus_count"), show_col_types = FALSE) %>%
  map(mutate, SRA = str_extract(SRA, "(SRR|ERR)[0-9]+")) %>%
  map(filter, str_starts(v_call, "IGH"), consensus_count > 2) %>%
  map(left_join, select(metadata, -Filename), by = c("SRA" = "Run")) %>%
  data.table::rbindlist() %>%
  as_tibble() %>%
  select(all_of(COLS))

bennett <- vroom(master_unfiltered, show_col_types = FALSE,
                 col_select = c(file_name, v_call, junction_aa, consensus_count, sequence, junction)) %>%
  filter(!str_detect(file_name, "SRR"), !str_detect(file_name, "ERR")) %>%
  filter(file_name %in% keep_one_file_per_run(unique(file_name))) %>%   # drops per-index copies
  filter(str_starts(v_call, "IGH"), consensus_count > 2) %>%
  left_join(filter(metadata, str_starts(BioProject, "Bennett")), by = c("file_name" = "Filename")) %>%
  filter(str_starts(BioProject, "Bennett"), !is.na(sequence), !is.na(junction)) %>%
  select(all_of(COLS))

# Written in two parts, as originally: in the 798 TRUST4 files with no assemblies (header only)
# consensus_count is read as text, so the two tables do not share a column type
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
write_csv(sra, out_csv)
write_csv(bennett, out_csv, append = TRUE)
cat(nrow(sra), "SRA rows +", nrow(bennett), "in-house rows written to", out_csv, "\n")
