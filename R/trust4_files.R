# One TRUST4 AIRR file per sequencing library.
# - SRA runs: TRUST4 wrote two files for 40 runs (the 10x-mode runs also have a per-cell
#   *_barcode_airr.tsv, and two qc-mode runs have an extra single-end *_<run>_airr.tsv next to the paired
#   *_<run>_1_airr.tsv). The all-contig, paired file is kept, as for every other run.
# - In-house: the ON07-05A_gc1 library also exists as 16 per-index files (*_gc1_AA_R1 ... *_gc1_TT_R1)
#   next to the combined *_gc1_R1 file. Per-index files are dropped when their combined file exists.
# Other in-house files are kept unchanged.

keep_one_file_per_run <- function(files) {
  name <- basename(files)
  run  <- stringr::str_extract(name, "(SRR|ERR)[0-9]+")
  ord  <- order(run, grepl("_barcode_", name), !grepl("_1_airr", name), name)
  extra <- logical(length(files))
  extra[ord] <- !is.na(run[ord]) & duplicated(run[ord])
  index_split <- grepl("_[ACGT]{2}_R1_airr\\.tsv$", name) &
    sub("_[ACGT]{2}_R1_airr\\.tsv$", "_R1_airr.tsv", name) %in% name
  files[!extra & !index_split]
}
