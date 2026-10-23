# V-J clipped pre/post treatment sequences (classifier/treatment_sequences_vj_{train,test}.csv), the
# second variant of Fig. 3B. build_treatment_dataset.Rmd keeps the full TRUST4 contig (leader/5' UTR and
# the start of the constant region included), while the classifiers were trained on IgBLAST V-J
# sequences (build_dataset.Rmd). This script annotates the contigs with IgBLAST and the same IMGT
# databases, keeps the rows that pass the classifier-dataset filters (productive, complete VDJ, IGH,
# no N in the clipped sequence) and replaces `sequence` with v_sequence_start..j_sequence_end.
#
# All original columns are kept. Added: `contig` (the TRUST4 sequence), `v_sequence_start` and
# `j_sequence_end` (IgBLAST, 1-based in `contig`) and `contig_row`, the row of the sequence in the
# contig predictions (treatment_predictions_{exclusion,baseline}.csv: train-fold rows first, then test).
#
# Usage:
#   Rscript dataset/09_treatment_vj.R <treatment_sequences_train.csv> <treatment_sequences_test.csv> \
#       <treatment_sequences_vj_train.csv> <treatment_sequences_vj_test.csv>

suppressPackageStartupMessages({library(tidyverse); library(vroom); library(igblastr)})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: Rscript dataset/09_treatment_vj.R <train_in> <test_in> <train_out> <test_out>")
THREADS <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))
CHUNK   <- 200000L   # sequences per igblastn call (bounds memory)

use_germline_db("IMGT-202531-1.Homo_sapiens.IGH+IGK+IGL")   # as in dataset/build_dataset.Rmd
use_c_region_db("_IMGT.human.IGH+IGK+IGL.202412")

# Read as text so every original column is written back unchanged
read_fold <- function(f) vroom(f, col_types = cols(.default = col_character()), show_col_types = FALSE)
folds <- list(train = read_fold(args[1]), test = read_fold(args[2]))
offset <- c(train = 0L, test = nrow(folds$train))

clip_fold <- function(df, first_row) {
  idx <- split(seq_len(nrow(df)), ceiling(seq_len(nrow(df)) / CHUNK))
  ig <- map_dfr(idx, function(i) {
    q <- Biostrings::DNAStringSet(df$sequence[i])
    names(q) <- i
    res <- igblastn(query = q, organism = "human", outfmt = "AIRR", num_threads = THREADS,
                    num_alignments_V = 1, num_alignments_C = 1)
    stopifnot(nrow(res) == length(i), all(as.integer(res$sequence_id) == i))
    cat(sprintf("  igblastn: %s of %s\n", scales::comma(max(i)), scales::comma(nrow(df))))
    select(res, productive, complete_vdj, ig_v_call = v_call, v_sequence_start, j_sequence_end)
  })
  out <- df %>%
    mutate(contig_row = first_row + row_number(), contig = sequence) %>%
    bind_cols(ig) %>%
    mutate(sequence = str_sub(contig, v_sequence_start, j_sequence_end))
  # Filter log, in the order build_dataset.Rmd applies them
  steps <- out %>%
    transmute(p = productive %in% TRUE, pc = p & complete_vdj %in% TRUE, pci = pc & str_starts(ig_v_call, "IGH") %in% TRUE) %>%
    summarize(input = n(), productive = sum(p), complete_vdj = sum(pc), IGH = sum(pci))
  out <- out %>%
    filter(productive == TRUE, complete_vdj == TRUE, str_starts(ig_v_call, "IGH")) %>%
    filter(str_count(sequence, "N") < 1)
  print(mutate(steps, no_N = nrow(out), kept = sprintf("%.1f%%", 100 * nrow(out) / input)))
  out %>% select(all_of(names(df)), contig, v_sequence_start, j_sequence_end, contig_row)
}

for (fold in names(folds)) {
  cat(sprintf("%s fold: %s sequences\n", fold, scales::comma(nrow(folds[[fold]]))))
  out <- clip_fold(folds[[fold]], offset[[fold]])
  out_csv <- args[if (fold == "train") 3 else 4]
  dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
  write_csv(out, out_csv, na = "NA")
  cat(sprintf("  -> %s (%s rows)\n", out_csv, scales::comma(nrow(out))))
}
