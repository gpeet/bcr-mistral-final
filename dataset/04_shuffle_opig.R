# Build the unselected pretraining corpus: IGH sequences from the master OPIG table, shuffled.
# Diversity selection (05_select_diverse_sequences.py) reads this file in order, and the
# random-sample ablation corpus is its first N lines (N = size of the diverse corpus).
#
# Usage:
#   Rscript dataset/04_shuffle_opig.R <master_opig_csv> <output_csv> [seed]
#
# Writes one sequence per line, no header. The published corpus was shuffled without a
# fixed seed, so a rebuild gives a different order (and a different diverse selection);
# the released diverse_foundation_data.csv is the corpus the models were trained on.

suppressPackageStartupMessages(library(tidyverse))

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% 2:3) stop("Usage: Rscript dataset/04_shuffle_opig.R <master_opig_csv> <output_csv> [seed]")
set.seed(if (length(args) == 3) as.integer(args[3]) else 42)

vroom::vroom(args[1], col_select = c('sequence', 'v_call')) %>%
  filter(str_starts(v_call, "IGH")) %>%
  select(sequence) %>%
  mutate(sequence = sample(sequence)) -> shuffled_opig

dir.create(dirname(args[2]), recursive = TRUE, showWarnings = FALSE)
write_csv(shuffled_opig, args[2], col_names = F)
cat(nrow(shuffled_opig), "sequences written to", args[2], "\n")
